#!/usr/bin/env python3
"""
Word-Level Beam Search Decoder for Azerbaijani Sign Language (AzSL).

Decodes a sequence of 32-class letter probability vectors into candidate words
using acoustic scores, bigram language model transitions, and lexicon prefix constraints.

Supports 4 decoding configurations:
  A. Greedy decoding (argmax at each position)
  B. Beam search without lexicon constraint (pure acoustic)
  C. Beam search + bigram language model
  D. Beam search + bigram language model + AZ_LEXICON prefix constraints
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# Canonical 32-letter Azerbaijani Latin alphabet
AZ_ALPHABET = [
    "A", "B", "C", "Ç", "D", "E", "Ə", "F", "G", "Ğ", "H", "X", "I", "İ",
    "J", "K", "Q", "L", "M", "N", "O", "Ö", "P", "R", "S", "Ş", "T", "U",
    "Ü", "V", "Y", "Z",
]
LETTER_TO_IDX = {ch: i for i, ch in enumerate(AZ_ALPHABET)}
IDX_TO_LETTER = {i: ch for i, ch in enumerate(AZ_ALPHABET)}

# Canonical MVP Lexicon matching index.html
AZ_LEXICON = [
    "MƏN", "SƏN", "O", "BİZ", "SİZ", "ONLAR", "NƏ", "KİM", "HARADA", "NECƏ", "NİYƏ", "HANSI",
    "SALAM", "SAĞOL", "XOŞ", "BƏLİ", "YOX", "BUYUR", "TƏŞƏKKÜR",
    "BİR", "İKİ", "ÜÇ", "DÖRD", "BEŞ", "ALTI", "YEDDİ", "SƏKKİZ", "DOQQUZ", "ON",
    "ANA", "ATA", "BACI", "QARDAŞ", "UŞAQ", "DOST",
    "EV", "BAKI", "GƏNCƏ", "ŞƏKİ", "ŞUŞA", "NAXÇIVAN", "ŞƏHƏR", "ÖLKƏ", "MƏKTƏB", "MEŞƏ", "DƏNİZ", "DAĞ",
    "SU", "ÇÖRƏK", "KİTAB", "QAPI", "MASA", "STUL", "MAŞIN", "PUL", "YOL", "GÜL", "AĞAC", "QUŞ",
    "İT", "PİŞİK", "BALIQ", "ÇAY", "SÜD", "ƏT", "ALMA", "ARMUD", "ÜZÜM",
    "GÜN", "GECƏ", "İL", "AY", "HƏFTƏ", "VAXT", "SƏHƏR", "AXŞAM",
    "YAXŞI", "PİS", "BÖYÜK", "KİÇİK", "GÖZƏL", "İSTİ", "SOYUQ", "YENİ", "ÇOX", "AZ",
    "GETMƏK", "GƏLMƏK", "GÖRMƏK", "BİLMƏK", "SEVMƏK", "İSTƏMƏK", "YAZMAQ", "OXUMAQ", "YEMƏK", "İÇMƏK", "İŞLƏMƏK", "DANIŞMAQ",
]


class TrieNode:
    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.is_word: bool = False


class LexiconTrie:
    """Fast prefix tree for dictionary lookup and candidate constraint."""

    def __init__(self, words: Optional[List[str]] = None):
        self.root = TrieNode()
        self.words: Set[str] = set()
        if words:
            for w in words:
                self.insert(w)

    def insert(self, word: str):
        word = word.strip().upper()
        if not word:
            return
        self.words.add(word)
        curr = self.root
        for ch in word:
            if ch not in curr.children:
                curr.children[ch] = TrieNode()
            curr = curr.children[ch]
        curr.is_word = True

    def is_valid_prefix(self, prefix: str) -> bool:
        prefix = prefix.strip().upper()
        curr = self.root
        for ch in prefix:
            if ch not in curr.children:
                return False
            curr = curr.children[ch]
        return True

    def is_valid_word(self, word: str) -> bool:
        word = word.strip().upper()
        return word in self.words

    def get_valid_continuations(self, prefix: str) -> List[str]:
        prefix = prefix.strip().upper()
        curr = self.root
        for ch in prefix:
            if ch not in curr.children:
                return []
            curr = curr.children[ch]
        return list(curr.children.keys())


class BigramLanguageModel:
    """Laplace-smoothed bigram language model over character sequences."""

    def __init__(self, corpus_words: List[str], alpha: float = 0.3):
        self.alpha = alpha
        self.alphabet_size = len(AZ_ALPHABET)
        self.bigram_counts: Dict[str, Dict[str, int]] = {}
        self.bigram_totals: Dict[str, int] = {}
        self.unigram_counts: Dict[str, int] = {}
        self.total_unigrams = 0

        # Fit model on lexicon word list
        for word in corpus_words:
            w = word.strip().upper()
            if not w:
                continue
            for i, ch in enumerate(w):
                self.unigram_counts[ch] = self.unigram_counts.get(ch, 0) + 1
                self.total_unigrams += 1
                if i > 0:
                    prev = w[i - 1]
                    if prev not in self.bigram_counts:
                        self.bigram_counts[prev] = {}
                    self.bigram_counts[prev][ch] = self.bigram_counts[prev].get(ch, 0) + 1
                    self.bigram_totals[prev] = self.bigram_totals.get(prev, 0) + 1

    def log_prob(self, curr: str, prev: Optional[str] = None) -> float:
        """Returns ln P(curr | prev) with Laplace smoothing."""
        if prev is None or prev not in self.bigram_totals:
            # Unigram prior with Laplace smoothing
            count = self.unigram_counts.get(curr, 0)
            p = (count + self.alpha) / (self.total_unigrams + self.alpha * self.alphabet_size)
            return math.log(max(p, 1e-12))

        count = self.bigram_counts.get(prev, {}).get(curr, 0)
        total = self.bigram_totals.get(prev, 0)
        p = (count + self.alpha) / (total + self.alpha * self.alphabet_size)
        return math.log(max(p, 1e-12))


@dataclass
class BeamHypothesis:
    sequence: str
    log_score: float
    acoustic_log_score: float
    lm_log_score: float
    is_valid_word: bool
    is_valid_prefix: bool

    def __repr__(self):
        status = " [WORD]" if self.is_valid_word else (" [PRE]" if self.is_valid_prefix else "")
        return f"Hypothesis('{self.sequence}', score={self.log_score:.3f}{status})"


class BeamSearchDecoder:
    """
    Configurable word-level beam search decoder.
    Scores candidates using acoustic evidence, bigram transitions, and lexicon constraints.
    """

    def __init__(
        self,
        beam_width: int = 5,
        lm_weight: float = 0.6,
        lexicon_word_bonus: float = 2.0,
        strict_lexicon_prefix: bool = True,
        invalid_prefix_penalty: float = 15.0,
        non_word_penalty: float = 5.0,
        lexicon_words: Optional[List[str]] = None,
    ):
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.lexicon_word_bonus = lexicon_word_bonus
        self.strict_lexicon_prefix = strict_lexicon_prefix
        self.invalid_prefix_penalty = invalid_prefix_penalty
        self.non_word_penalty = non_word_penalty

        words = lexicon_words or AZ_LEXICON
        self.trie = LexiconTrie(words)
        self.lm = BigramLanguageModel(words, alpha=0.3)

    def decode(
        self,
        prob_sequence: List[np.ndarray],
        beam_width: Optional[int] = None,
        use_lm: bool = True,
        use_lexicon: bool = True,
    ) -> List[BeamHypothesis]:
        """
        Decodes a sequence of probability vectors.
        prob_sequence: List of 32-dim probability arrays, one per timestep.
        """
        B = beam_width if beam_width is not None else self.beam_width
        if not prob_sequence:
            return []

        # Initial beam: empty string with score 0.0
        # Stores tuples: (sequence, log_score, acoustic_log, lm_log)
        beam: List[Tuple[str, float, float, float]] = [("", 0.0, 0.0, 0.0)]

        for t, prob_vec in enumerate(prob_sequence):
            candidates: List[Tuple[str, float, float, float]] = []

            # Prune observation vector to top candidates to avoid evaluating near-zero logits
            top_token_indices = np.argsort(prob_vec)[::-1][:min(12, len(AZ_ALPHABET))]

            for seq, cum_score, ac_score, lm_score in beam:
                prev_char = seq[-1] if len(seq) > 0 else None

                for token_idx in top_token_indices:
                    char = IDX_TO_LETTER[token_idx]
                    p_ac = float(prob_vec[token_idx])
                    if p_ac <= 1e-12:
                        continue

                    new_seq = seq + char
                    new_ac_score = ac_score + math.log(p_ac)

                    # Language model score
                    lm_delta = self.lm.log_prob(char, prev_char) if use_lm else 0.0
                    new_lm_score = lm_score + lm_delta

                    # Lexicon constraint check
                    is_pre = self.trie.is_valid_prefix(new_seq)
                    lex_penalty = 0.0
                    if use_lexicon:
                        if not is_pre:
                            if self.strict_lexicon_prefix:
                                continue  # Prune directly
                            lex_penalty = -self.invalid_prefix_penalty

                    # Combined score
                    score = new_ac_score + (self.lm_weight * new_lm_score) + lex_penalty
                    candidates.append((new_seq, score, new_ac_score, new_lm_score))

            if not candidates:
                # Fallback in case strict lexicon pruned everything: pick top acoustic token
                best_token = IDX_TO_LETTER[int(np.argmax(prob_vec))]
                for seq, cum_score, ac_score, lm_score in beam:
                    new_seq = seq + best_token
                    p_ac = float(prob_vec[LETTER_TO_IDX[best_token]])
                    candidates.append((new_seq, cum_score + math.log(max(p_ac, 1e-12)), ac_score + math.log(max(p_ac, 1e-12)), lm_score))

            # Select top B candidates for next timestep
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:B]

        # Final word-level scoring at T
        hypotheses: List[BeamHypothesis] = []
        for seq, score, ac_score, lm_score in beam:
            is_word = self.trie.is_valid_word(seq)
            is_pre = self.trie.is_valid_prefix(seq)

            final_score = score
            if use_lexicon:
                if is_word:
                    final_score += self.lexicon_word_bonus
                elif not self.strict_lexicon_prefix:
                    final_score -= self.non_word_penalty

            hypotheses.append(
                BeamHypothesis(
                    sequence=seq,
                    log_score=final_score,
                    acoustic_log_score=ac_score,
                    lm_log_score=lm_score,
                    is_valid_word=is_word,
                    is_valid_prefix=is_pre,
                )
            )

        hypotheses.sort(key=lambda h: h.log_score, reverse=True)
        return hypotheses

    def greedy_decode(self, prob_sequence: List[np.ndarray]) -> Tuple[str, float]:
        """Greedy argmax baseline decoding."""
        seq = ""
        log_prob = 0.0
        for vec in prob_sequence:
            idx = int(np.argmax(vec))
            p = float(vec[idx])
            seq += IDX_TO_LETTER[idx]
            log_prob += math.log(max(p, 1e-12))
        return seq, log_prob


def create_simulated_sequence(
    target_word: str,
    ambiguities: Optional[Dict[int, Tuple[str, float, float]]] = None,
    noise_std: float = 0.02,
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Constructs a controlled sequence of 32-class probability vectors for a target word.
    target_word: e.g. "BAKI"
    ambiguities: Dict mapping index -> (competitor_char, competitor_prob, target_prob)
                 e.g. {0: ("R", 0.45, 0.38)} -> at position 0, 'R' has higher prob than 'B' (greedy failure)
    """
    rng = np.random.RandomState(seed)
    seq_probs = []

    for t, target_char in enumerate(target_word.upper()):
        target_idx = LETTER_TO_IDX[target_char]
        probs = np.full(len(AZ_ALPHABET), 0.001, dtype=np.float64)

        if ambiguities and t in ambiguities:
            comp_char, comp_prob, targ_prob = ambiguities[t]
            comp_idx = LETTER_TO_IDX[comp_char]
            rem = max(0.001, 1.0 - comp_prob - targ_prob)
            probs[:] = rem / len(AZ_ALPHABET)
            probs[comp_idx] = comp_prob
            probs[target_idx] = targ_prob
        else:
            # Strong clear acoustic detection (0.90 target confidence)
            rem = 1.0 - 0.90
            probs[:] = rem / (len(AZ_ALPHABET) - 1)
            probs[target_idx] = 0.90

        probs /= np.sum(probs)
        seq_probs.append(probs)

    return seq_probs


def run_controlled_experiments(beam_widths: List[int] = [3, 5, 10]) -> Dict[str, Any]:
    """Runs evaluation across the 4 decoding configurations and multiple beam widths."""
    decoder = BeamSearchDecoder(lm_weight=0.6, lexicon_word_bonus=3.0)

    # Define test suite: (Word, Ambiguity Description, Ambiguity Dict)
    test_suite = [
        # Ambiguous cases designed to break greedy decoding
        ("BAKI", "Pos 0: R(46%) > B(38%) [Greedy fails]", {0: ("R", 0.46, 0.38)}),
        ("SALAM", "Pos 3: Ə(48%) > A(39%) [Greedy fails]", {3: ("Ə", 0.48, 0.39)}),
        ("GÜL", "Pos 0: Ş(49%) > G(39%) [Greedy fails (G vs Ş)]", {0: ("Ş", 0.49, 0.39)}),
        ("SƏHƏR", "Pos 3: P(47%) > H(37%) [Greedy fails (H vs P)]", {3: ("P", 0.47, 0.37)}),
        ("KİTAB", "Pos 0: Ç(45%) > K(38%) [Greedy fails]", {0: ("Ç", 0.45, 0.38)}),
        ("DƏNİZ", "Pos 0: K(45%) > D(38%) [Greedy fails]", {0: ("K", 0.45, 0.38)}),
        ("ÇÖRƏK", "Pos 4: T(46%) > K(39%) [Greedy fails]", {4: ("T", 0.46, 0.39)}),
        ("DOST", "Pos 2: L(48%) > S(39%) [Greedy fails]", {2: ("L", 0.48, 0.39)}),
        # Clean sequences (Greedy already correct)
        ("BİZ", "Clean acoustic sequence", None),
        ("ANA", "Clean acoustic sequence", None),
        ("ALMA", "Clean acoustic sequence", None),
        ("ŞƏKİ", "Clean acoustic sequence", None),
    ]

    configs = [
        ("A. Greedy", "greedy", False, False),
        ("B. Beam (Acoustic Only)", "beam", False, False),
        ("C. Beam + Bigram LM", "beam", True, False),
        ("D. Beam + Bigram LM + AZ_LEXICON", "beam", True, True),
    ]

    all_results = []

    for word, desc, amb in test_suite:
        seq = create_simulated_sequence(word, ambiguities=amb, seed=42)
        greedy_pred, greedy_score = decoder.greedy_decode(seq)
        greedy_correct = (greedy_pred == word)

        case_record = {
            "target_word": word,
            "description": desc,
            "has_ambiguity": amb is not None,
            "greedy_pred": greedy_pred,
            "greedy_score": round(greedy_score, 3),
            "greedy_correct": greedy_correct,
            "beam_runs": {},
        }

        for bw in beam_widths:
            case_record["beam_runs"][bw] = {}
            for cfg_name, mode, use_lm, use_lex in configs:
                if mode == "greedy":
                    pred = greedy_pred
                    score = greedy_score
                    is_corr = greedy_correct
                    rescued = False
                else:
                    t0 = time.perf_counter()
                    hyps = decoder.decode(seq, beam_width=bw, use_lm=use_lm, use_lexicon=use_lex)
                    dur = (time.perf_counter() - t0) * 1000.0
                    top_hyp = hyps[0] if hyps else None
                    pred = top_hyp.sequence if top_hyp else ""
                    score = top_hyp.log_score if top_hyp else -999.0
                    is_corr = (pred == word)
                    rescued = (not greedy_correct and is_corr)

                case_record["beam_runs"][bw][cfg_name] = {
                    "pred": pred,
                    "score": round(score, 3),
                    "correct": is_corr,
                    "rescued": rescued,
                }

        all_results.append(case_record)

    return {"results": all_results, "beam_widths": beam_widths, "configs": [c[0] for c in configs]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam-widths", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument("--output", default="models/beam_decoder_eval_report.json")
    args = parser.parse_args()

    print(f"Running Controlled Beam Search Decoding Experiments across beam widths {args.beam_widths}...\n")
    exp = run_controlled_experiments(beam_widths=args.beam_widths)

    print("=" * 95)
    print("  PHASE 1 & 2: CONTROLLED DECODER EVALUATION ON AZERBAIJANI WORDS")
    print("=" * 95)

    header = f"{'Target':<8}{'Ambiguity Scenario':<36}{'Config A (Greedy)':<18}{'Config C (+Bigram)':<18}{'Config D (+Lexicon)'}"
    print(header)
    print("-" * 95)

    total_cases = len(exp["results"])
    accuracy_by_cfg = {cfg: 0 for cfg in exp["configs"]}
    rescues_by_cfg = {cfg: 0 for cfg in exp["configs"]}

    for item in exp["results"]:
        target = item["target_word"]
        desc = item["description"]
        # Use B=5 for the representative printout
        runs_b5 = item["beam_runs"][5]

        pred_a = runs_b5["A. Greedy"]["pred"]
        pred_c = runs_b5["C. Beam + Bigram LM"]["pred"]
        pred_d = runs_b5["D. Beam + Bigram LM + AZ_LEXICON"]["pred"]

        mark_a = " [OK]" if pred_a == target else " [ERR]"
        mark_c = " [OK]" if pred_c == target else " [ERR]"
        mark_d = " [OK]" if pred_d == target else " [ERR]"

        print(f"{target:<8}{desc:<36}{pred_a + mark_a:<18}{pred_c + mark_c:<18}{pred_d + mark_d}")

        for cfg in exp["configs"]:
            if runs_b5[cfg]["correct"]:
                accuracy_by_cfg[cfg] += 1
            if runs_b5[cfg]["rescued"]:
                rescues_by_cfg[cfg] += 1

    print("\n" + "=" * 95)
    print("  SUMMARY PERFORMANCE COMPARISON ACROSS CONFIGURATIONS (Beam Width = 5)")
    print("=" * 95)
    print(f"{'Configuration':<38}{'Accuracy':<14}{'Word Error Rate':<18}{'Greedy Errors Rescued'}")
    print("-" * 95)

    for cfg in exp["configs"]:
        acc = accuracy_by_cfg[cfg] / total_cases * 100.0
        wer = 100.0 - acc
        rescued = rescues_by_cfg[cfg]
        print(f"{cfg:<38}{acc:>6.1f}%        {wer:>6.1f}%            {rescued} rescued")
    print("-" * 95)

    # Beam Width Impact Comparison
    print("\n" + "=" * 95)
    print("  IMPACT OF BEAM WIDTH (B = 3 vs. B = 5 vs. B = 10) on Config D (+Lexicon)")
    print("=" * 95)
    print(f"{'Beam Width':<16}{'Accuracy':<14}{'Total Words Evaluated':<24}{'Errors Corrected'}")
    print("-" * 95)
    for bw in args.beam_widths:
        bw_acc = sum(1 for r in exp["results"] if r["beam_runs"][bw]["D. Beam + Bigram LM + AZ_LEXICON"]["correct"])
        bw_rescues = sum(1 for r in exp["results"] if r["beam_runs"][bw]["D. Beam + Bigram LM + AZ_LEXICON"]["rescued"])
        pct = (bw_acc / total_cases) * 100.0
        print(f"B = {bw:<12}{pct:>6.1f}%        {total_cases:<24}{bw_rescues} errors corrected")
    print("-" * 95)

    # Latency benchmarking
    decoder = BeamSearchDecoder(lexicon_words=AZ_LEXICON)
    sample_seq = create_simulated_sequence("TƏŞƏKKÜR")
    print("\nDecoder Latency Benchmark (1,000 runs on 8-letter word 'TƏŞƏKKÜR'):")
    for bw in args.beam_widths:
        t0 = time.perf_counter()
        for _ in range(1000):
            decoder.decode(sample_seq, beam_width=bw, use_lm=True, use_lexicon=True)
        avg_ms = ((time.perf_counter() - t0) / 1000.0) * 1000.0
        print(f"  • Beam Width B={bw:<2}: {avg_ms:.3f} ms per word")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(exp, f, ensure_ascii=False, indent=2)
    print(f"\nFull evaluation report saved to: {args.output}")


if __name__ == "__main__":
    main()

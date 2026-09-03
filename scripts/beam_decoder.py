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
        lexicon_word_bonus: float = 3.0,
        lexicon_mode: str = "soft",  # 'none', 'hard', 'soft'
        strict_lexicon_prefix: Optional[bool] = None,
        lexicon_words: Optional[List[str]] = None,
    ):
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.lexicon_word_bonus = lexicon_word_bonus
        if strict_lexicon_prefix is not None:
            self.lexicon_mode = "hard" if strict_lexicon_prefix else "soft"
        else:
            self.lexicon_mode = lexicon_mode

        words = lexicon_words or AZ_LEXICON
        self.trie = LexiconTrie(words)
        self.lm = BigramLanguageModel(words, alpha=0.3)

    def decode(
        self,
        prob_sequence: List[np.ndarray],
        beam_width: Optional[int] = None,
        use_lm: bool = True,
        use_lexicon: Optional[bool] = None,
        lexicon_mode: Optional[str] = None,
        lexicon_word_bonus: Optional[float] = None,
    ) -> List[BeamHypothesis]:
        """
        Decodes a sequence of probability vectors.
        prob_sequence: List of 32-dim probability arrays, one per timestep.
        lexicon_mode: 'none', 'hard' (strict prefix pruning), or 'soft' (positive bonus at word end).
        """
        B = beam_width if beam_width is not None else self.beam_width
        bonus = lexicon_word_bonus if lexicon_word_bonus is not None else self.lexicon_word_bonus

        if lexicon_mode is not None:
            active_lex_mode = lexicon_mode
        elif use_lexicon is False:
            active_lex_mode = "none"
        elif use_lexicon is True:
            active_lex_mode = self.lexicon_mode if self.lexicon_mode != "none" else "soft"
        else:
            active_lex_mode = self.lexicon_mode

        if not prob_sequence:
            return []

        # Initial beam: empty string with score 0.0
        # Stores tuples: (sequence, log_score, acoustic_log, lm_log)
        beam: List[Tuple[str, float, float, float]] = [("", 0.0, 0.0, 0.0)]

        for t, prob_vec in enumerate(prob_sequence):
            candidates: List[Tuple[str, float, float, float]] = []

            # Prune observation vector to top candidates
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
                    # In HARD mode: prune if prefix is invalid
                    # In SOFT or NONE mode: NEVER prune based on lexicon prefix
                    if active_lex_mode == "hard":
                        if not self.trie.is_valid_prefix(new_seq):
                            continue

                    # Combined acoustic + language model score
                    score = new_ac_score + (self.lm_weight * new_lm_score)
                    candidates.append((new_seq, score, new_ac_score, new_lm_score))

            if not candidates:
                # Fallback in case hard lexicon pruned everything: pick top acoustic token
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
            # In SOFT and HARD modes: valid lexicon words receive a positive score bonus
            if active_lex_mode in ("soft", "hard") and is_word:
                final_score += bonus

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
    seed: int = 42,
) -> List[np.ndarray]:
    """Constructs a controlled sequence of 32-class probability vectors for a target word."""
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
            rem = 1.0 - 0.90
            probs[:] = rem / (len(AZ_ALPHABET) - 1)
            probs[target_idx] = 0.90

        probs /= np.sum(probs)
        seq_probs.append(probs)

    return seq_probs


def main():
    print("BeamSearchDecoder module loaded successfully.")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Experimental Evaluation: Old 70-word Bigram Corpus vs. New 19,734-word Lexicon Corpus.

Evaluates on the exact same 150 real sequences (75 IV + 75 OOV) across:
  - Scenario 1: Clean real dataset samples
  - Scenario 2: Live camera variation (rotation + jitter)

Compares:
  A. Greedy Decoding (Baseline)
  B. Beam + Bigram using OLD 70-word corpus (beta=0)
  C. Beam + Bigram using NEW 19,734-word corpus (beta=0)
  D. Beam + Bigram (New 19k corpus) + Soft Lexicon (beta=3.0 on 70-word AZ_LEXICON)

Reports metrics and specifically checks whether any OOV words degrade.
Saves results to:
  models/real_pipeline_lexicon_comparison_report.json
  models/real_pipeline_lexicon_comparison_results.csv
"""

import argparse
import csv
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.beam_decoder import (
    AZ_ALPHABET,
    AZ_LEXICON,
    LETTER_TO_IDX,
    IDX_TO_LETTER,
    BeamSearchDecoder,
    load_lexicon_corpus,
)
from scripts.static_model import StaticHierarchicalModel, augment_landmarks, build_feature_vector_84


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculates Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


IV_WORDS = [
    "BAKI", "SALAM", "GÜL", "DƏNİZ", "KİTAB",
    "ANA", "BİZ", "ŞƏHƏR", "DOST", "ALMA",
    "ÇÖRƏK", "PUL", "SU", "EV", "GÖZƏL",
]

OOV_WORDS = [
    "QƏLƏM", "BAHAR", "SƏMA", "BULUD", "GƏMİ",
    "HƏYAT", "VƏTƏN", "BAYRAQ", "ÇANTA", "DƏFTƏR",
    "PƏNCƏRƏ", "MƏKTUB", "BULAQ", "YAĞIŞ", "QUMLUQ",
]


def build_real_sequence(
    target_word: str,
    samples_by_letter: Dict[str, List[Dict[str, Any]]],
    model: StaticHierarchicalModel,
    sample_index: int = 0,
    apply_jitter: bool = False,
) -> Tuple[List[np.ndarray], List[str]]:
    prob_sequence = []
    sample_ids = []

    for t, ch in enumerate(target_word.upper()):
        available = samples_by_letter.get(ch, [])
        if not available:
            raise ValueError(f"No samples available in cache for letter '{ch}'")

        rec = available[sample_index % len(available)]
        sample_ids.append(f"{ch}_{sample_index % len(available)}")

        if apply_jitter:
            rng = np.random.RandomState(sample_index * 100 + t)
            aug_coords = augment_landmarks(rec["coords"], max_angles=(10.0, 10.0, 12.0), jitter_std=0.015, rng=rng)
            feat84 = build_feature_vector_84(aug_coords, np.zeros(2))
        else:
            feat84 = rec["feat84"]

        res = model.predict_from_feature_vector(feat84, mode="soft", top_k=2)
        prob_dict = res.get("distribution", {})
        prob_vec = np.zeros(len(AZ_ALPHABET), dtype=np.float64)

        floor_prob = 1e-5
        prob_vec[:] = floor_prob

        for letter, p in prob_dict.items():
            if letter in LETTER_TO_IDX:
                prob_vec[LETTER_TO_IDX[letter]] = max(p, floor_prob)

        prob_vec /= np.sum(prob_vec)
        prob_sequence.append(prob_vec)

    return prob_sequence, sample_ids


def evaluate_lexicon_comparison(
    words: List[str],
    is_iv: bool,
    samples_per_word: int,
    samples_by_letter: Dict[str, List[Dict[str, Any]]],
    model: StaticHierarchicalModel,
    decoder_old_lm: BeamSearchDecoder,
    decoder_new_lm: BeamSearchDecoder,
    beam_width: int = 5,
    apply_jitter: bool = False,
) -> List[Dict[str, Any]]:
    records = []

    for word in words:
        for s_idx in range(samples_per_word):
            prob_seq, sample_ids = build_real_sequence(
                word, samples_by_letter, model, sample_index=s_idx, apply_jitter=apply_jitter
            )

            # A. Greedy
            t0 = time.perf_counter()
            pred_a, score_a = decoder_old_lm.greedy_decode(prob_seq)
            lat_a = (time.perf_counter() - t0) * 1000.0
            dist_a = levenshtein_distance(word, pred_a)
            corr_a = (pred_a == word)
            match_a = sum(1 for c1, c2 in zip(word, pred_a) if c1 == c2) if len(pred_a) == len(word) else max(0, len(word) - dist_a)

            # B. Beam + Bigram (OLD 70-word corpus, beta=0)
            t0 = time.perf_counter()
            hyps_b = decoder_old_lm.decode(prob_seq, beam_width=beam_width, use_lm=True, lexicon_mode="none")
            lat_b = (time.perf_counter() - t0) * 1000.0
            pred_b = hyps_b[0].sequence if hyps_b else ""
            dist_b = levenshtein_distance(word, pred_b)
            corr_b = (pred_b == word)
            match_b = sum(1 for c1, c2 in zip(word, pred_b) if c1 == c2) if len(pred_b) == len(word) else max(0, len(word) - dist_b)

            # C. Beam + Bigram (NEW 19,734-word corpus, beta=0)
            t0 = time.perf_counter()
            hyps_c = decoder_new_lm.decode(prob_seq, beam_width=beam_width, use_lm=True, lexicon_mode="none")
            lat_c = (time.perf_counter() - t0) * 1000.0
            pred_c = hyps_c[0].sequence if hyps_c else ""
            dist_c = levenshtein_distance(word, pred_c)
            corr_c = (pred_c == word)
            match_c = sum(1 for c1, c2 in zip(word, pred_c) if c1 == c2) if len(pred_c) == len(word) else max(0, len(word) - dist_c)

            # D. Beam + Bigram (NEW 19,734-word corpus) + Soft Lexicon (beta=3.0 on 70-word AZ_LEXICON)
            t0 = time.perf_counter()
            hyps_d = decoder_new_lm.decode(prob_seq, beam_width=beam_width, use_lm=True, lexicon_mode="soft", lexicon_word_bonus=3.0)
            lat_d = (time.perf_counter() - t0) * 1000.0
            pred_d = hyps_d[0].sequence if hyps_d else ""
            dist_d = levenshtein_distance(word, pred_d)
            corr_d = (pred_d == word)
            match_d = sum(1 for c1, c2 in zip(word, pred_d) if c1 == c2) if len(pred_d) == len(word) else max(0, len(word) - dist_d)

            records.append({
                "target_word": word,
                "word_length": len(word),
                "is_in_vocabulary": is_iv,
                "apply_jitter": apply_jitter,
                "sample_index": s_idx,
                "sample_ids": sample_ids,
                # Config A: Greedy
                "pred_a": pred_a,
                "correct_a": corr_a,
                "char_matches_a": match_a,
                "edit_dist_a": dist_a,
                "latency_ms_a": round(lat_a, 4),
                # Config B: Beam + Bigram (Old 70 words)
                "pred_b": pred_b,
                "correct_b": corr_b,
                "char_matches_b": match_b,
                "edit_dist_b": dist_b,
                "rescued_b": (not corr_a and corr_b),
                "hurt_b": (corr_a and not corr_b),
                "latency_ms_b": round(lat_b, 4),
                # Config C: Beam + Bigram (New 19k words)
                "pred_c": pred_c,
                "correct_c": corr_c,
                "char_matches_c": match_c,
                "edit_dist_c": dist_c,
                "rescued_c": (not corr_a and corr_c),
                "hurt_c": (corr_a and not corr_c),
                "latency_ms_c": round(lat_c, 4),
                # Config D: Beam + Bigram (New 19k) + Soft Lexicon (beta=3.0)
                "pred_d": pred_d,
                "correct_d": corr_d,
                "char_matches_d": match_d,
                "edit_dist_d": dist_d,
                "rescued_d": (not corr_a and corr_d),
                "hurt_d": (corr_a and not corr_d),
                "latency_ms_d": round(lat_d, 4),
                # Direct comparison: Did New 19k change prediction from Old 70?
                "old_to_new_changed": (pred_b != pred_c),
                "new_improved_over_old": (not corr_b and corr_c),
                "new_degraded_from_old": (corr_b and not corr_c),
            })

    return records


def compute_metrics(records: List[Dict[str, Any]], pred_key: str, corr_key: str, match_key: str, dist_key: str, lat_key: str) -> Dict[str, Any]:
    n_words = len(records)
    total_chars = sum(r["word_length"] for r in records)
    if n_words == 0:
        return {}

    word_acc = sum(1 for r in records if r[corr_key]) / n_words
    char_acc = sum(r[match_key] for r in records) / total_chars
    wer = sum(1 for r in records if not r[corr_key]) / n_words
    cer = sum(r[dist_key] for r in records) / total_chars
    rescued = sum(1 for r in records if (not r["correct_a"] and r[corr_key]))
    hurt = sum(1 for r in records if (r["correct_a"] and not r[corr_key]))
    avg_lat = np.mean([r[lat_key] for r in records])

    return {
        "word_acc": round(word_acc * 100, 2),
        "char_acc": round(char_acc * 100, 2),
        "wer": round(wer * 100, 2),
        "cer": round(cer * 100, 2),
        "rescued_count": rescued,
        "hurt_count": hurt,
        "avg_latency_ms": round(float(avg_lat), 4),
    }


def analyze_all_configurations(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "A_greedy": compute_metrics(records, "pred_a", "correct_a", "char_matches_a", "edit_dist_a", "latency_ms_a"),
        "B_beam_old_70": compute_metrics(records, "pred_b", "correct_b", "char_matches_b", "edit_dist_b", "latency_ms_b"),
        "C_beam_new_19k": compute_metrics(records, "pred_c", "correct_c", "char_matches_c", "edit_dist_c", "latency_ms_c"),
        "D_beam_new_19k_soft_lex": compute_metrics(records, "pred_d", "correct_d", "char_matches_d", "edit_dist_d", "latency_ms_d"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexicon-path", default="lexicon.txt")
    parser.add_argument("--cache-path", default="models/.static_landmarks_cache.pkl")
    parser.add_argument("--model-path", default="public/models/azsl_hierarchical_model.json")
    parser.add_argument("--samples-per-word", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--json-output", default="models/real_pipeline_lexicon_comparison_report.json")
    parser.add_argument("--csv-output", default="models/real_pipeline_lexicon_comparison_results.csv")
    args = parser.parse_args()

    print(f"Loading new lexicon corpus from {args.lexicon_path}...")
    new_corpus_words = load_lexicon_corpus(args.lexicon_path)
    print(f"Loaded {len(new_corpus_words)} word tokens ({len(set(new_corpus_words))} unique words) from {args.lexicon_path}.")

    print("Loading pre-trained StaticHierarchicalModel...")
    model = StaticHierarchicalModel(args.model_path)

    print("Loading real landmark cache...")
    with open(args.cache_path, "rb") as f:
        cache_data = pickle.load(f)

    samples_by_letter: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r, letter in zip(cache_data["records"], cache_data["y_letter"]):
        samples_by_letter[letter].append(r)
    print(f"Loaded {len(cache_data['records'])} records across {len(samples_by_letter)} letters.")

    # Instantiate decoders
    decoder_old_lm = BeamSearchDecoder(
        beam_width=args.beam_width,
        lm_weight=0.6,
        lexicon_word_bonus=3.0,
        lexicon_mode="soft",
        lexicon_words=AZ_LEXICON,
        bigram_corpus=AZ_LEXICON,  # OLD 70-word corpus
    )

    decoder_new_lm = BeamSearchDecoder(
        beam_width=args.beam_width,
        lm_weight=0.6,
        lexicon_word_bonus=3.0,
        lexicon_mode="soft",
        lexicon_words=AZ_LEXICON,  # 70-word lexicon for membership bonus
        bigram_corpus=new_corpus_words,  # NEW 19,734-word corpus for bigrams
    )

    # -------------------------------------------------------------
    # Scenario 1: Clean Real Dataset Samples
    # -------------------------------------------------------------
    print("\nEvaluating Scenario 1: Clean Real Dataset Samples (N=150)...")
    iv_clean = evaluate_lexicon_comparison(IV_WORDS, True, args.samples_per_word, samples_by_letter, model, decoder_old_lm, decoder_new_lm, args.beam_width, False)
    oov_clean = evaluate_lexicon_comparison(OOV_WORDS, False, args.samples_per_word, samples_by_letter, model, decoder_old_lm, decoder_new_lm, args.beam_width, False)
    all_clean = iv_clean + oov_clean

    clean_iv_m = analyze_all_configurations(iv_clean)
    clean_oov_m = analyze_all_configurations(oov_clean)
    clean_all_m = analyze_all_configurations(all_clean)

    # -------------------------------------------------------------
    # Scenario 2: Real Samples under Live Camera Variation
    # -------------------------------------------------------------
    print("Evaluating Scenario 2: Live Camera Variation (Rotation + Jitter) (N=150)...")
    iv_jitter = evaluate_lexicon_comparison(IV_WORDS, True, args.samples_per_word, samples_by_letter, model, decoder_old_lm, decoder_new_lm, args.beam_width, True)
    oov_jitter = evaluate_lexicon_comparison(OOV_WORDS, False, args.samples_per_word, samples_by_letter, model, decoder_old_lm, decoder_new_lm, args.beam_width, True)
    all_jitter = iv_jitter + oov_jitter

    jitter_iv_m = analyze_all_configurations(iv_jitter)
    jitter_oov_m = analyze_all_configurations(oov_jitter)
    jitter_all_m = analyze_all_configurations(all_jitter)

    # Print Formatted Comparison Tables
    def print_table(title: str, m_iv: Dict[str, Any], m_oov: Dict[str, Any], m_all: Dict[str, Any]):
        print("\n" + "=" * 115)
        print(f"  {title}")
        print("=" * 115)
        print(f"{'Configuration':<38}{'IV Word Acc':<14}{'OOV Word Acc':<15}{'All Word Acc':<15}{'All WER':<10}{'All CER':<10}{'Rescued':<10}{'Hurt'}")
        print("-" * 115)

        cfgs = [
            ("A_greedy", "A. Greedy (Baseline)"),
            ("B_beam_old_70", "B. Beam + Bigram (Old 70-word LM)"),
            ("C_beam_new_19k", "C. Beam + Bigram (New 19k-word LM)"),
            ("D_beam_new_19k_soft_lex", "D. Beam + New LM + Soft Lexicon"),
        ]

        for key, name in cfgs:
            iv = m_iv[key]
            oov = m_oov[key]
            comb = m_all[key]
            r_str = str(comb['rescued_count']) if key != "A_greedy" else "-"
            h_str = str(comb['hurt_count']) if key != "A_greedy" else "-"
            print(f"{name:<38}{iv['word_acc']:>6.2f}%        {oov['word_acc']:>6.2f}%         {comb['word_acc']:>6.2f}%        {comb['wer']:>5.2f}%    {comb['cer']:>5.2f}%    {r_str:>5s}     {h_str:>5s}")
        print("-" * 115)

    print_table("SCENARIO 1: CLEAN REAL DATASET SAMPLES (N=150: 75 IV + 75 OOV)", clean_iv_m, clean_oov_m, clean_all_m)
    print_table("SCENARIO 2: LIVE CAMERA VARIATION (N=150: 75 IV + 75 OOV)", jitter_iv_m, jitter_oov_m, jitter_all_m)

    # Detailed differences between Old 70-word LM vs New 19k-word LM on live variation
    print("\n" + "=" * 115)
    print("  DETAILED COMPARISON: OLD 70-WORD BIGRAM VS. NEW 19,734-WORD BIGRAM (Live Variation Scenario)")
    print("=" * 115)
    changes = [r for r in all_jitter if r["old_to_new_changed"]]
    print(f"Total sequences with different predictions between Old LM and New LM: {len(changes)} / {len(all_jitter)}")

    improved = [r for r in all_jitter if r["new_improved_over_old"]]
    degraded = [r for r in all_jitter if r["new_degraded_from_old"]]

    print(f"  • Improved by New 19k LM (rescued from Old LM error) : {len(improved)}")
    for r in improved:
        print(f"    - Target: {r['target_word']:<10} (IV={r['is_in_vocabulary']}) | Greedy: {r['pred_a']:<10} | Old LM: {r['pred_b']:<10} -> New LM: {r['pred_c']}")

    print(f"  • Degraded by New 19k LM (hurt previously correct Old LM) : {len(degraded)}")
    for r in degraded:
        print(f"    - Target: {r['target_word']:<10} (IV={r['is_in_vocabulary']}) | Greedy: {r['pred_a']:<10} | Old LM: {r['pred_b']:<10} -> New LM: {r['pred_c']}")

    # Check specifically for OOV words
    oov_changes = [r for r in oov_jitter if r["old_to_new_changed"]]
    oov_degraded = [r for r in oov_jitter if r["new_degraded_from_old"]]
    print(f"\n  • Total OOV words changed by New 19k LM : {len(oov_changes)}")
    print(f"  • Total OOV words DEGRADED by New 19k LM: {len(oov_degraded)}")
    if oov_degraded:
        for r in oov_degraded:
            print(f"    * DEGRADED OOV Word: {r['target_word']} | Old LM: {r['pred_b']} -> New LM: {r['pred_c']}")
    else:
        print("    * ZERO OOV words degraded! Open-vocabulary recognition completely intact.")

    # Save to JSON
    os.makedirs(os.path.dirname(args.json_output), exist_ok=True)
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "lexicon_corpus_path": args.lexicon_path,
            "lexicon_total_tokens": len(new_corpus_words),
            "lexicon_unique_words": len(set(new_corpus_words)),
            "total_sequences_per_scenario": len(all_clean),
            "iv_sequences_count": len(iv_clean),
            "oov_sequences_count": len(oov_clean),
            "samples_per_word": args.samples_per_word,
            "beam_width": args.beam_width,
            "scenario_1_clean": {
                "iv_metrics": clean_iv_m,
                "oov_metrics": clean_oov_m,
                "overall_metrics": clean_all_m,
            },
            "scenario_2_live_variation": {
                "iv_metrics": jitter_iv_m,
                "oov_metrics": jitter_oov_m,
                "overall_metrics": jitter_all_m,
                "changed_sequences_count": len(changes),
                "improved_sequences_count": len(improved),
                "degraded_sequences_count": len(degraded),
                "oov_degraded_count": len(oov_degraded),
            },
        },
    }

    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"\nReport JSON saved to: {args.json_output}")

    # Save to CSV
    os.makedirs(os.path.dirname(args.csv_output), exist_ok=True)
    csv_rows = []
    for r in all_clean + all_jitter:
        csv_rows.append({
            "target_word": r["target_word"],
            "word_length": r["word_length"],
            "is_in_vocabulary": r["is_in_vocabulary"],
            "apply_jitter": r["apply_jitter"],
            "sample_index": r["sample_index"],
            "pred_greedy": r["pred_a"],
            "correct_greedy": r["correct_a"],
            "pred_old_70_lm": r["pred_b"],
            "correct_old_70_lm": r["correct_b"],
            "pred_new_19k_lm": r["pred_c"],
            "correct_new_19k_lm": r["correct_c"],
            "pred_new_19k_soft_lex": r["pred_d"],
            "correct_new_19k_soft_lex": r["correct_d"],
            "new_improved": r["new_improved_over_old"],
            "new_degraded": r["new_degraded_from_old"],
        })

    with open(args.csv_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Raw results CSV saved to: {args.csv_output}")


if __name__ == "__main__":
    main()


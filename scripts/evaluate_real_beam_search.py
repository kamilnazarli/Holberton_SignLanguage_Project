#!/usr/bin/env python3
"""
Evaluation of Word Beam Search on Real Pipeline Outputs.

Compares:
  A. Greedy Letter-by-Letter Decoding (Baseline)
  B. Beam Search + Bigram Language Model
  C. Beam Search + Bigram Language Model + AZ_LEXICON Constraints

Evaluated on REAL 32-class probability distributions produced by StaticHierarchicalModel
(soft routing K=2, exactly matching the browser runtime) across real landmark samples.

Separates performance into:
  1. In-Vocabulary (IV) words present in AZ_LEXICON
  2. Out-of-Vocabulary (OOV) words NOT present in AZ_LEXICON
"""

import argparse
import csv
import json
import math
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
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
)
from scripts.static_model import StaticHierarchicalModel


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


# 15 In-Vocabulary words (present in AZ_LEXICON)
IV_WORDS = [
    "BAKI", "SALAM", "GÜL", "DƏNİZ", "KİTAB",
    "ANA", "BİZ", "ŞƏHƏR", "DOST", "ALMA",
    "ÇÖRƏK", "PUL", "SU", "EV", "GÖZƏL",
]

# 15 Out-of-Vocabulary words (valid Azerbaijani words deliberately NOT in AZ_LEXICON)
OOV_WORDS = [
    "QƏLƏM", "BAHAR", "SƏMA", "BULUD", "GƏMİ",
    "HƏYAT", "VƏTƏN", "BAYRAQ", "ÇANTA", "DƏFTƏR",
    "PƏNCƏRƏ", "MƏKTUB", "BULAQ", "YAĞIŞ", "QUMLUQ",
]

CONFUSION_PAIRS = [
    ("G", "Ş"),
    ("H", "P"),
    ("B", "R"),
    ("C", "K"),
    ("A", "Ə"),
]


def build_real_sequence(
    target_word: str,
    samples_by_letter: Dict[str, List[Dict[str, Any]]],
    model: StaticHierarchicalModel,
    sample_index: int = 0,
    apply_jitter: bool = False,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Constructs a sequence of REAL 32-class probability distributions for a target word
    by evaluating the actual static model on real dataset landmark samples.
    """
    from scripts.static_model import augment_landmarks, build_feature_vector_84

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

        # Build full 32-class probability vector
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


def evaluate_decoders(
    words: List[str],
    is_iv: bool,
    samples_per_word: int,
    samples_by_letter: Dict[str, List[Dict[str, Any]]],
    model: StaticHierarchicalModel,
    decoder: BeamSearchDecoder,
    beam_width: int = 5,
    apply_jitter: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluates all sample sequences across Decoders A, B, and C."""
    # Soft decoder for comparative study (does not prune non-lexicon prefixes)
    soft_lex_decoder = BeamSearchDecoder(
        beam_width=beam_width,
        lm_weight=0.6,
        lexicon_word_bonus=3.0,
        strict_lexicon_prefix=False,
    )
    records = []

    for word in words:
        for s_idx in range(samples_per_word):
            prob_seq, sample_ids = build_real_sequence(
                word, samples_by_letter, model, sample_index=s_idx, apply_jitter=apply_jitter
            )

            # 1. Decoder A: Greedy (Letter-by-Letter baseline)
            t0 = time.perf_counter()
            pred_a, score_a = decoder.greedy_decode(prob_seq)
            lat_a = (time.perf_counter() - t0) * 1000.0

            # 2. Decoder B: Beam + Bigram LM
            t0 = time.perf_counter()
            hyps_b = decoder.decode(prob_seq, beam_width=beam_width, use_lm=True, use_lexicon=False)
            lat_b = (time.perf_counter() - t0) * 1000.0
            top_b = hyps_b[0] if hyps_b else None
            pred_b = top_b.sequence if top_b else ""
            score_b = top_b.log_score if top_b else -999.0

            # 3. Decoder C: Beam + Bigram LM + AZ_LEXICON (Strict Prefix)
            t0 = time.perf_counter()
            hyps_c = decoder.decode(prob_seq, beam_width=beam_width, use_lm=True, use_lexicon=True)
            lat_c = (time.perf_counter() - t0) * 1000.0
            top_c = hyps_c[0] if hyps_c else None
            pred_c = top_c.sequence if top_c else ""
            score_c = top_c.log_score if top_c else -999.0

            # 4. Decoder D: Beam + Bigram LM + AZ_LEXICON (Soft Bonus)
            t0 = time.perf_counter()
            hyps_d = soft_lex_decoder.decode(prob_seq, beam_width=beam_width, use_lm=True, use_lexicon=True)
            lat_d = (time.perf_counter() - t0) * 1000.0
            top_d = hyps_d[0] if hyps_d else None
            pred_d = top_d.sequence if top_d else ""
            score_d = top_d.log_score if top_d else -999.0

            # Compute error metrics
            dist_a = levenshtein_distance(word, pred_a)
            dist_b = levenshtein_distance(word, pred_b)
            dist_c = levenshtein_distance(word, pred_c)
            dist_d = levenshtein_distance(word, pred_d)

            match_chars_a = sum(1 for c1, c2 in zip(word, pred_a) if c1 == c2) if len(pred_a) == len(word) else max(0, len(word) - dist_a)
            match_chars_b = sum(1 for c1, c2 in zip(word, pred_b) if c1 == c2) if len(pred_b) == len(word) else max(0, len(word) - dist_b)
            match_chars_c = sum(1 for c1, c2 in zip(word, pred_c) if c1 == c2) if len(pred_c) == len(word) else max(0, len(word) - dist_c)
            match_chars_d = sum(1 for c1, c2 in zip(word, pred_d) if c1 == c2) if len(pred_d) == len(word) else max(0, len(word) - dist_d)

            corr_a = (pred_a == word)
            corr_b = (pred_b == word)
            corr_c = (pred_c == word)
            corr_d = (pred_d == word)

            record = {
                "target_word": word,
                "word_length": len(word),
                "is_in_vocabulary": is_iv,
                "apply_jitter": apply_jitter,
                "sample_index": s_idx,
                "sample_ids": sample_ids,
                # Decoder A (Greedy)
                "pred_a": pred_a,
                "score_a": round(score_a, 3),
                "correct_a": corr_a,
                "char_matches_a": match_chars_a,
                "edit_dist_a": dist_a,
                "latency_ms_a": round(lat_a, 4),
                # Decoder B (Beam + Bigram)
                "pred_b": pred_b,
                "score_b": round(score_b, 3),
                "correct_b": corr_b,
                "char_matches_b": match_chars_b,
                "edit_dist_b": dist_b,
                "latency_ms_b": round(lat_b, 4),
                "rescued_b": (not corr_a and corr_b),
                "hurt_b": (corr_a and not corr_b),
                # Decoder C (Beam + Bigram + Lexicon Strict)
                "pred_c": pred_c,
                "score_c": round(score_c, 3),
                "correct_c": corr_c,
                "char_matches_c": match_chars_c,
                "edit_dist_c": dist_c,
                "latency_ms_c": round(lat_c, 4),
                "rescued_c": (not corr_a and corr_c),
                "hurt_c": (corr_a and not corr_c),
                # Decoder D (Beam + Bigram + Lexicon Soft Bonus)
                "pred_d": pred_d,
                "score_d": round(score_d, 3),
                "correct_d": corr_d,
                "char_matches_d": match_chars_d,
                "edit_dist_d": dist_d,
                "latency_ms_d": round(lat_d, 4),
                "rescued_d": (not corr_a and corr_d),
                "hurt_d": (corr_a and not corr_d),
            }
            records.append(record)

    return records


def compute_aggregate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates overall Character Accuracy, Word Accuracy, CER, WER, Rescued, and Hurt."""
    n_words = len(records)
    total_chars = sum(r["word_length"] for r in records)

    if n_words == 0:
        return {}

    # Decoder A
    word_acc_a = sum(1 for r in records if r["correct_a"]) / n_words
    char_acc_a = sum(r["char_matches_a"] for r in records) / total_chars
    cer_a = sum(r["edit_dist_a"] for r in records) / total_chars
    wer_a = sum(1 for r in records if not r["correct_a"]) / n_words
    lat_a = np.mean([r["latency_ms_a"] for r in records])

    # Decoder B
    word_acc_b = sum(1 for r in records if r["correct_b"]) / n_words
    char_acc_b = sum(r["char_matches_b"] for r in records) / total_chars
    cer_b = sum(r["edit_dist_b"] for r in records) / total_chars
    wer_b = sum(1 for r in records if not r["correct_b"]) / n_words
    rescued_b = sum(1 for r in records if r["rescued_b"])
    hurt_b = sum(1 for r in records if r["hurt_b"])
    lat_b = np.mean([r["latency_ms_b"] for r in records])

    # Decoder C
    word_acc_c = sum(1 for r in records if r["correct_c"]) / n_words
    char_acc_c = sum(r["char_matches_c"] for r in records) / total_chars
    cer_c = sum(r["edit_dist_c"] for r in records) / total_chars
    wer_c = sum(1 for r in records if not r["correct_c"]) / n_words
    rescued_c = sum(1 for r in records if r["rescued_c"])
    hurt_c = sum(1 for r in records if r["hurt_c"])
    lat_c = np.mean([r["latency_ms_c"] for r in records])

    return {
        "n_words": n_words,
        "total_chars": total_chars,
        "decoder_a_greedy": {
            "word_acc": round(word_acc_a * 100, 2),
            "char_acc": round(char_acc_a * 100, 2),
            "wer": round(wer_a * 100, 2),
            "cer": round(cer_a * 100, 2),
            "avg_latency_ms": round(float(lat_a), 4),
        },
        "decoder_b_beam_bigram": {
            "word_acc": round(word_acc_b * 100, 2),
            "char_acc": round(char_acc_b * 100, 2),
            "wer": round(wer_b * 100, 2),
            "cer": round(cer_b * 100, 2),
            "rescued_count": rescued_b,
            "hurt_count": hurt_b,
            "avg_latency_ms": round(float(lat_b), 4),
        },
        "decoder_c_beam_lexicon": {
            "word_acc": round(word_acc_c * 100, 2),
            "char_acc": round(char_acc_c * 100, 2),
            "wer": round(wer_c * 100, 2),
            "cer": round(cer_c * 100, 2),
            "rescued_count": rescued_c,
            "hurt_count": hurt_c,
            "avg_latency_ms": round(float(lat_c), 4),
        },
    }


def analyze_confusions(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Inspects errors specifically involving target confusion pairs (G/Ş, H/P, B/R, C/K, A/Ə)."""
    confusion_stats = {}

    for c1, c2 in CONFUSION_PAIRS:
        pair_key = f"{c1} vs {c2}"
        confusion_stats[pair_key] = {
            "occurrences_in_reference": 0,
            "greedy_errors": 0,
            "beam_bigram_errors": 0,
            "beam_lexicon_errors": 0,
            "detailed_examples": [],
        }

        for r in records:
            ref = r["target_word"]
            for idx, ch in enumerate(ref):
                if ch in (c1, c2):
                    confusion_stats[pair_key]["occurrences_in_reference"] += 1
                    comp = c2 if ch == c1 else c1

                    pred_a = r["pred_a"][idx] if idx < len(r["pred_a"]) else ""
                    pred_b = r["pred_b"][idx] if idx < len(r["pred_b"]) else ""
                    pred_c = r["pred_c"][idx] if idx < len(r["pred_c"]) else ""

                    err_a = (pred_a == comp)
                    err_b = (pred_b == comp)
                    err_c = (pred_c == comp)

                    if err_a:
                        confusion_stats[pair_key]["greedy_errors"] += 1
                    if err_b:
                        confusion_stats[pair_key]["beam_bigram_errors"] += 1
                    if err_c:
                        confusion_stats[pair_key]["beam_lexicon_errors"] += 1

                    if err_a or err_b or err_c:
                        confusion_stats[pair_key]["detailed_examples"].append({
                            "word": ref,
                            "pos": idx,
                            "target": ch,
                            "pred_a": pred_a,
                            "pred_b": pred_b,
                            "pred_c": pred_c,
                        })

    return confusion_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", default="models/.static_landmarks_cache.pkl")
    parser.add_argument("--model-path", default="public/models/azsl_hierarchical_model.json")
    parser.add_argument("--samples-per-word", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--json-output", default="models/real_pipeline_beam_eval_report.json")
    parser.add_argument("--csv-output", default="models/real_pipeline_beam_eval_results.csv")
    args = parser.parse_args()

    print("Loading pre-trained StaticHierarchicalModel...")
    model = StaticHierarchicalModel(args.model_path)

    print("Loading real landmark cache...")
    with open(args.cache_path, "rb") as f:
        cache_data = pickle.load(f)

    # Index real records by letter
    samples_by_letter: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r, letter in zip(cache_data["records"], cache_data["y_letter"]):
        samples_by_letter[letter].append(r)

    print(f"Loaded {len(cache_data['records'])} records across {len(samples_by_letter)} letters.")
    decoder = BeamSearchDecoder(beam_width=args.beam_width, lm_weight=0.6, lexicon_word_bonus=3.0)

    # -------------------------------------------------------------
    # Scenario 1: Clean Real Dataset Samples (No noise)
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print("  SCENARIO 1: CLEAN REAL DATASET SAMPLES (Exact Cached Landmarks)")
    print("=" * 95)
    iv_clean = evaluate_decoders(IV_WORDS, True, args.samples_per_word, samples_by_letter, model, decoder, args.beam_width, apply_jitter=False)
    oov_clean = evaluate_decoders(OOV_WORDS, False, args.samples_per_word, samples_by_letter, model, decoder, args.beam_width, apply_jitter=False)
    all_clean = iv_clean + oov_clean

    iv_clean_m = compute_aggregate_metrics(iv_clean)
    oov_clean_m = compute_aggregate_metrics(oov_clean)
    all_clean_m = compute_aggregate_metrics(all_clean)
    conf_clean = analyze_confusions(all_clean)

    def print_metric_table(title: str, m: Dict[str, Any]):
        print(f"\n--- {title} ---")
        print(f"{'Decoder Configuration':<38}{'Word Acc':<12}{'Char Acc':<12}{'WER':<10}{'CER':<10}{'Rescued':<10}{'Hurt'}")
        print("-" * 98)
        da = m["decoder_a_greedy"]
        print(f"{'A. Greedy (Baseline)':<38}{da['word_acc']:>6.2f}%      {da['char_acc']:>6.2f}%     {da['wer']:>5.2f}%    {da['cer']:>5.2f}%        -         -")
        db = m["decoder_b_beam_bigram"]
        print(f"{'B. Beam + Bigram LM':<38}{db['word_acc']:>6.2f}%      {db['char_acc']:>6.2f}%     {db['wer']:>5.2f}%    {db['cer']:>5.2f}%     {db['rescued_count']:>4d}      {db['hurt_count']:>4d}")
        dc = m["decoder_c_beam_lexicon"]
        print(f"{'C. Beam + Bigram + AZ_LEXICON (Strict)':<38}{dc['word_acc']:>6.2f}%      {dc['char_acc']:>6.2f}%     {dc['wer']:>5.2f}%    {dc['cer']:>5.2f}%     {dc['rescued_count']:>4d}      {dc['hurt_count']:>4d}")
        print("-" * 98)

    print_metric_table("1A. In-Vocabulary Words (Clean, N=75)", iv_clean_m)
    print_metric_table("1B. Out-of-Vocabulary Words (Clean, N=75)", oov_clean_m)
    print_metric_table("1C. Combined Benchmark (Clean, N=150)", all_clean_m)

    # -------------------------------------------------------------
    # Scenario 2: Real Samples under Realistic Live Camera Jitter
    # -------------------------------------------------------------
    print("\n" + "=" * 95)
    print("  SCENARIO 2: REAL SAMPLES UNDER REALISTIC LIVE CAMERA VARIATION (Rotation & Jitter)")
    print("=" * 95)
    iv_jitter = evaluate_decoders(IV_WORDS, True, args.samples_per_word, samples_by_letter, model, decoder, args.beam_width, apply_jitter=True)
    oov_jitter = evaluate_decoders(OOV_WORDS, False, args.samples_per_word, samples_by_letter, model, decoder, args.beam_width, apply_jitter=True)
    all_jitter = iv_jitter + oov_jitter

    iv_jitter_m = compute_aggregate_metrics(iv_jitter)
    oov_jitter_m = compute_aggregate_metrics(oov_jitter)
    all_jitter_m = compute_aggregate_metrics(all_jitter)
    conf_jitter = analyze_confusions(all_jitter)

    print_metric_table("2A. In-Vocabulary Words (Live Variation, N=75)", iv_jitter_m)
    print_metric_table("2B. Out-of-Vocabulary Words (Live Variation, N=75)", oov_jitter_m)
    print_metric_table("2C. Combined Benchmark (Live Variation, N=150)", all_jitter_m)

    # Confusion Analysis Table
    print("\n--- 3. FOCUSED CONFUSION PAIR ANALYSIS (Live Variation Scenario) ---")
    print(f"{'Confusion Pair':<18}{'Total In Ref':<16}{'Greedy Errs':<16}{'Beam+Bigram Errs':<20}{'Beam+Lexicon Errs'}")
    print("-" * 90)
    for pair, stats in conf_jitter.items():
        print(f"{pair:<18}{stats['occurrences_in_reference']:>8d}        {stats['greedy_errors']:>8d}        {stats['beam_bigram_errors']:>10d}          {stats['beam_lexicon_errors']:>10d}")
    print("-" * 90)

    # Average Latencies
    print("\n--- 4. INFERENCE LATENCY BENCHMARK ---")
    print(f"  • Decoder A (Greedy)             : {all_clean_m['decoder_a_greedy']['avg_latency_ms']:.4f} ms per word")
    print(f"  • Decoder B (Beam + Bigram)      : {all_clean_m['decoder_b_beam_bigram']['avg_latency_ms']:.4f} ms per word")
    print(f"  • Decoder C (Beam + Lexicon)     : {all_clean_m['decoder_c_beam_lexicon']['avg_latency_ms']:.4f} ms per word")

    # Save to JSON
    os.makedirs(os.path.dirname(args.json_output), exist_ok=True)
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "total_words_evaluated_per_scenario": len(all_clean),
            "iv_words_count": len(iv_clean),
            "oov_words_count": len(oov_clean),
            "samples_per_word": args.samples_per_word,
            "beam_width": args.beam_width,
            "scenario_1_clean": {
                "iv_metrics": iv_clean_m,
                "oov_metrics": oov_clean_m,
                "overall_metrics": all_clean_m,
                "confusion_analysis": conf_clean,
            },
            "scenario_2_live_variation": {
                "iv_metrics": iv_jitter_m,
                "oov_metrics": oov_jitter_m,
                "overall_metrics": all_jitter_m,
                "confusion_analysis": conf_jitter,
            },
        },
        "clean_records": all_clean,
        "jitter_records": all_jitter,
    }

    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"\nFull report JSON saved to: {args.json_output}")

    # Save to CSV
    os.makedirs(os.path.dirname(args.csv_output), exist_ok=True)
    csv_fields = [
        "target_word", "word_length", "is_in_vocabulary", "apply_jitter", "sample_index",
        "pred_a", "correct_a", "edit_dist_a", "latency_ms_a",
        "pred_b", "correct_b", "edit_dist_b", "rescued_b", "hurt_b", "latency_ms_b",
        "pred_c", "correct_c", "edit_dist_c", "rescued_c", "hurt_c", "latency_ms_c",
    ]
    with open(args.csv_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_clean + all_jitter:
            writer.writerow(r)
    print(f"Raw results CSV saved to: {args.csv_output}")


if __name__ == "__main__":
    main()

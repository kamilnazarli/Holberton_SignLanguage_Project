#!/usr/bin/env python3
"""
Comprehensive Evaluation of Soft Lexicon Word Beam Search.

Compares:
  A. Greedy Decoding (Baseline)
  B. Beam Search + Bigram LM (no lexicon, beta=0)
  C. Beam Search + Bigram LM + Hard Lexicon (strict prefix pruning, beta=3.0)
  D. Beam Search + Bigram LM + Soft Lexicon tested across beta in [0.0, 0.5, 1.0, 2.0, 3.0]

Evaluated on the exact same 150 real sequences (75 IV + 75 OOV) across:
  - Scenario 1: Clean real dataset samples
  - Scenario 2: Real samples under realistic live camera variation (rotation + jitter)
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

BETA_VALUES = [0.0, 0.5, 1.0, 2.0, 3.0]


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


def evaluate_all_configurations(
    words: List[str],
    is_iv: bool,
    samples_per_word: int,
    samples_by_letter: Dict[str, List[Dict[str, Any]]],
    model: StaticHierarchicalModel,
    decoder: BeamSearchDecoder,
    beam_width: int = 5,
    apply_jitter: bool = False,
) -> List[Dict[str, Any]]:
    records = []

    for word in words:
        for s_idx in range(samples_per_word):
            prob_seq, sample_ids = build_real_sequence(
                word, samples_by_letter, model, sample_index=s_idx, apply_jitter=apply_jitter
            )

            # 1. Configuration A: Greedy
            t0 = time.perf_counter()
            pred_a, score_a = decoder.greedy_decode(prob_seq)
            lat_a = (time.perf_counter() - t0) * 1000.0
            dist_a = levenshtein_distance(word, pred_a)
            corr_a = (pred_a == word)
            match_a = sum(1 for c1, c2 in zip(word, pred_a) if c1 == c2) if len(pred_a) == len(word) else max(0, len(word) - dist_a)

            # 2. Configuration B: Beam + Bigram (no lexicon)
            t0 = time.perf_counter()
            hyps_b = decoder.decode(prob_seq, beam_width=beam_width, use_lm=True, lexicon_mode="none")
            lat_b = (time.perf_counter() - t0) * 1000.0
            pred_b = hyps_b[0].sequence if hyps_b else ""
            dist_b = levenshtein_distance(word, pred_b)
            corr_b = (pred_b == word)
            match_b = sum(1 for c1, c2 in zip(word, pred_b) if c1 == c2) if len(pred_b) == len(word) else max(0, len(word) - dist_b)

            # 3. Configuration C: Beam + Bigram + Hard Lexicon (beta=3.0, strict prefix pruning)
            t0 = time.perf_counter()
            hyps_c = decoder.decode(prob_seq, beam_width=beam_width, use_lm=True, lexicon_mode="hard", lexicon_word_bonus=3.0)
            lat_c = (time.perf_counter() - t0) * 1000.0
            pred_c = hyps_c[0].sequence if hyps_c else ""
            dist_c = levenshtein_distance(word, pred_c)
            corr_c = (pred_c == word)
            match_c = sum(1 for c1, c2 in zip(word, pred_c) if c1 == c2) if len(pred_c) == len(word) else max(0, len(word) - dist_c)

            rec = {
                "target_word": word,
                "word_length": len(word),
                "is_in_vocabulary": is_iv,
                "apply_jitter": apply_jitter,
                "sample_index": s_idx,
                "sample_ids": sample_ids,
                # A: Greedy
                "pred_a": pred_a,
                "correct_a": corr_a,
                "char_matches_a": match_a,
                "edit_dist_a": dist_a,
                "latency_ms_a": round(lat_a, 4),
                # B: Beam + Bigram
                "pred_b": pred_b,
                "correct_b": corr_b,
                "char_matches_b": match_b,
                "edit_dist_b": dist_b,
                "rescued_b": (not corr_a and corr_b),
                "hurt_b": (corr_a and not corr_b),
                "latency_ms_b": round(lat_b, 4),
                # C: Beam + Hard Lexicon
                "pred_c": pred_c,
                "correct_c": corr_c,
                "char_matches_c": match_c,
                "edit_dist_c": dist_c,
                "rescued_c": (not corr_a and corr_c),
                "hurt_c": (corr_a and not corr_c),
                "latency_ms_c": round(lat_c, 4),
                # D: Soft Lexicon across Beta sweep
                "soft_lexicon_betas": {},
            }

            # 4. Configuration D: Soft Lexicon across multiple beta values
            for beta in BETA_VALUES:
                t0 = time.perf_counter()
                hyps_d = decoder.decode(prob_seq, beam_width=beam_width, use_lm=True, lexicon_mode="soft", lexicon_word_bonus=beta)
                lat_d = (time.perf_counter() - t0) * 1000.0
                pred_d = hyps_d[0].sequence if hyps_d else ""
                dist_d = levenshtein_distance(word, pred_d)
                corr_d = (pred_d == word)
                match_d = sum(1 for c1, c2 in zip(word, pred_d) if c1 == c2) if len(pred_d) == len(word) else max(0, len(word) - dist_d)

                b_key = f"beta_{beta:.1f}"
                rec["soft_lexicon_betas"][b_key] = {
                    "pred": pred_d,
                    "correct": corr_d,
                    "char_matches": match_d,
                    "edit_dist": dist_d,
                    "rescued": (not corr_a and corr_d),
                    "hurt": (corr_a and not corr_d),
                    "latency_ms": round(lat_d, 4),
                }

            records.append(rec)

    return records


def compute_config_metrics(records: List[Dict[str, Any]], config_key: str) -> Dict[str, Any]:
    n_words = len(records)
    total_chars = sum(r["word_length"] for r in records)
    if n_words == 0:
        return {}

    if config_key == "greedy":
        word_acc = sum(1 for r in records if r["correct_a"]) / n_words
        char_acc = sum(r["char_matches_a"] for r in records) / total_chars
        cer = sum(r["edit_dist_a"] for r in records) / total_chars
        wer = sum(1 for r in records if not r["correct_a"]) / n_words
        rescued = 0
        hurt = 0
        avg_lat = np.mean([r["latency_ms_a"] for r in records])
    elif config_key == "beam_bigram":
        word_acc = sum(1 for r in records if r["correct_b"]) / n_words
        char_acc = sum(r["char_matches_b"] for r in records) / total_chars
        cer = sum(r["edit_dist_b"] for r in records) / total_chars
        wer = sum(1 for r in records if not r["correct_b"]) / n_words
        rescued = sum(1 for r in records if r["rescued_b"])
        hurt = sum(1 for r in records if r["hurt_b"])
        avg_lat = np.mean([r["latency_ms_b"] for r in records])
    elif config_key == "beam_hard_lexicon":
        word_acc = sum(1 for r in records if r["correct_c"]) / n_words
        char_acc = sum(r["char_matches_c"] for r in records) / total_chars
        cer = sum(r["edit_dist_c"] for r in records) / total_chars
        wer = sum(1 for r in records if not r["correct_c"]) / n_words
        rescued = sum(1 for r in records if r["rescued_c"])
        hurt = sum(1 for r in records if r["hurt_c"])
        avg_lat = np.mean([r["latency_ms_c"] for r in records])
    elif config_key.startswith("soft_"):
        b_key = config_key.replace("soft_", "")
        word_acc = sum(1 for r in records if r["soft_lexicon_betas"][b_key]["correct"]) / n_words
        char_acc = sum(r["soft_lexicon_betas"][b_key]["char_matches"] for r in records) / total_chars
        cer = sum(r["soft_lexicon_betas"][b_key]["edit_dist"] for r in records) / total_chars
        wer = sum(1 for r in records if not r["soft_lexicon_betas"][b_key]["correct"]) / n_words
        rescued = sum(1 for r in records if r["soft_lexicon_betas"][b_key]["rescued"])
        hurt = sum(1 for r in records if r["soft_lexicon_betas"][b_key]["hurt"])
        avg_lat = np.mean([r["soft_lexicon_betas"][b_key]["latency_ms"] for r in records])
    else:
        raise ValueError(f"Unknown config_key: {config_key}")

    return {
        "word_acc": round(word_acc * 100, 2),
        "char_acc": round(char_acc * 100, 2),
        "wer": round(wer * 100, 2),
        "cer": round(cer * 100, 2),
        "rescued_count": rescued,
        "hurt_count": hurt,
        "avg_latency_ms": round(float(avg_lat), 4),
    }


def analyze_all_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    configs = ["greedy", "beam_bigram", "beam_hard_lexicon"] + [f"soft_beta_{b:.1f}" for b in BETA_VALUES]
    metrics = {}
    for cfg in configs:
        metrics[cfg] = compute_config_metrics(records, cfg)
    return metrics


def analyze_oov_words_detail(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Inspects recognition outcomes specifically for the 15 requested OOV words."""
    oov_tracking = {}
    for word in OOV_WORDS:
        word_recs = [r for r in records if r["target_word"] == word]
        oov_tracking[word] = {
            "samples_evaluated": len(word_recs),
            "greedy_correct": sum(1 for r in word_recs if r["correct_a"]),
            "beam_bigram_correct": sum(1 for r in word_recs if r["correct_b"]),
            "hard_lexicon_correct": sum(1 for r in word_recs if r["correct_c"]),
            "soft_lexicon_correct": {
                f"beta_{b:.1f}": sum(1 for r in word_recs if r["soft_lexicon_betas"][f"beta_{b:.1f}"]["correct"])
                for b in BETA_VALUES
            },
            "hard_lexicon_predictions": [r["pred_c"] for r in word_recs],
            "soft_lexicon_predictions_beta1": [r["soft_lexicon_betas"]["beta_1.0"]["pred"] for r in word_recs],
        }
    return oov_tracking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", default="models/.static_landmarks_cache.pkl")
    parser.add_argument("--model-path", default="public/models/azsl_hierarchical_model.json")
    parser.add_argument("--samples-per-word", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--json-output", default="models/soft_lexicon_eval_report.json")
    parser.add_argument("--csv-output", default="models/soft_lexicon_eval_results.csv")
    args = parser.parse_args()

    print("Loading pre-trained StaticHierarchicalModel...")
    model = StaticHierarchicalModel(args.model_path)

    print("Loading real landmark cache...")
    with open(args.cache_path, "rb") as f:
        cache_data = pickle.load(f)

    samples_by_letter: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r, letter in zip(cache_data["records"], cache_data["y_letter"]):
        samples_by_letter[letter].append(r)

    print(f"Loaded {len(cache_data['records'])} records across {len(samples_by_letter)} letters.")
    decoder = BeamSearchDecoder(beam_width=args.beam_width, lm_weight=0.6, lexicon_word_bonus=3.0, lexicon_mode="soft")

    # -------------------------------------------------------------
    # Scenario 1: Clean Real Dataset Samples
    # -------------------------------------------------------------
    print("\nEvaluating Scenario 1: Clean Real Dataset Samples...")
    iv_clean = evaluate_all_configurations(IV_WORDS, True, args.samples_per_word, samples_by_letter, model, decoder, args.beam_width, False)
    oov_clean = evaluate_all_configurations(OOV_WORDS, False, args.samples_per_word, samples_by_letter, model, decoder, args.beam_width, False)
    all_clean = iv_clean + oov_clean

    clean_iv_m = analyze_all_metrics(iv_clean)
    clean_oov_m = analyze_all_metrics(oov_clean)
    clean_all_m = analyze_all_metrics(all_clean)
    clean_oov_details = analyze_oov_words_detail(oov_clean)

    # -------------------------------------------------------------
    # Scenario 2: Real Samples under Realistic Live Camera Variation
    # -------------------------------------------------------------
    print("Evaluating Scenario 2: Live Camera Variation (Rotation + Jitter)...")
    iv_jitter = evaluate_all_configurations(IV_WORDS, True, args.samples_per_word, samples_by_letter, model, decoder, args.beam_width, True)
    oov_jitter = evaluate_all_configurations(OOV_WORDS, False, args.samples_per_word, samples_by_letter, model, decoder, args.beam_width, True)
    all_jitter = iv_jitter + oov_jitter

    jitter_iv_m = analyze_all_metrics(iv_jitter)
    jitter_oov_m = analyze_all_metrics(oov_jitter)
    jitter_all_m = analyze_all_metrics(all_jitter)
    jitter_oov_details = analyze_oov_words_detail(oov_jitter)

    # Print Formatted Comparison Tables
    def print_comparison_table(title: str, m_iv: Dict[str, Any], m_oov: Dict[str, Any], m_all: Dict[str, Any]):
        print("\n" + "=" * 115)
        print(f"  {title}")
        print("=" * 115)
        print(f"{'Configuration':<34}{'IV Word Acc':<14}{'OOV Word Acc':<15}{'All Word Acc':<15}{'All WER':<10}{'All CER':<10}{'Rescued':<10}{'Hurt'}")
        print("-" * 115)

        names = [
            ("greedy", "A. Greedy (Baseline)"),
            ("beam_bigram", "B. Beam + Bigram (No Lexicon)"),
            ("beam_hard_lexicon", "C. Beam + Hard Lexicon (Pruning)"),
            ("soft_beta_0.0", "D. Soft Lexicon (beta=0.0)"),
            ("soft_beta_0.5", "D. Soft Lexicon (beta=0.5)"),
            ("soft_beta_1.0", "D. Soft Lexicon (beta=1.0)"),
            ("soft_beta_2.0", "D. Soft Lexicon (beta=2.0)"),
            ("soft_beta_3.0", "D. Soft Lexicon (beta=3.0)"),
        ]

        for key, display_name in names:
            iv = m_iv[key]
            oov = m_oov[key]
            comb = m_all[key]
            resc_str = str(comb['rescued_count']) if key != "greedy" else "-"
            hurt_str = str(comb['hurt_count']) if key != "greedy" else "-"
            print(f"{display_name:<34}{iv['word_acc']:>6.2f}%        {oov['word_acc']:>6.2f}%         {comb['word_acc']:>6.2f}%        {comb['wer']:>5.2f}%    {comb['cer']:>5.2f}%    {resc_str:>5s}     {hurt_str:>5s}")
        print("-" * 115)

    print_comparison_table("SCENARIO 1: CLEAN REAL DATASET SAMPLES (N=150: 75 IV + 75 OOV)", clean_iv_m, clean_oov_m, clean_all_m)
    print_comparison_table("SCENARIO 2: REAL SAMPLES UNDER LIVE CAMERA VARIATION (N=150: 75 IV + 75 OOV)", jitter_iv_m, jitter_oov_m, jitter_all_m)

    # Detailed 15 OOV Words Table
    print("\n" + "=" * 110)
    print("  DETAILED TRACKING: 15 SPECIFIC OUT-OF-VOCABULARY (OOV) WORDS (Live Variation Scenario)")
    print("=" * 110)
    print(f"{'Target Word':<12}{'Greedy':<10}{'Beam+Bigram':<14}{'Hard Lexicon':<15}{'Soft (beta=1)':<16}{'Hard Lexicon Distortion Example'}")
    print("-" * 110)
    for word in OOV_WORDS:
        d = jitter_oov_details[word]
        gr_str = f"{d['greedy_correct']}/5"
        bb_str = f"{d['beam_bigram_correct']}/5"
        hl_str = f"{d['hard_lexicon_correct']}/5"
        sl_str = f"{d['soft_lexicon_correct']['beta_1.0']}/5"
        distorted = [p for p in d['hard_lexicon_predictions'] if p != word]
        dist_sample = distorted[0] if distorted else "None (Correct)"
        print(f"{word:<12}{gr_str:<10}{bb_str:<14}{hl_str:<15}{sl_str:<16}{dist_sample}")
    print("-" * 110)

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
            "beta_values_tested": BETA_VALUES,
            "scenario_1_clean": {
                "iv_metrics": clean_iv_m,
                "oov_metrics": clean_oov_m,
                "overall_metrics": clean_all_m,
                "oov_details": clean_oov_details,
            },
            "scenario_2_live_variation": {
                "iv_metrics": jitter_iv_m,
                "oov_metrics": jitter_oov_m,
                "overall_metrics": jitter_all_m,
                "oov_details": jitter_oov_details,
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
        row = {
            "target_word": r["target_word"],
            "word_length": r["word_length"],
            "is_in_vocabulary": r["is_in_vocabulary"],
            "apply_jitter": r["apply_jitter"],
            "sample_index": r["sample_index"],
            "pred_a": r["pred_a"],
            "correct_a": r["correct_a"],
            "pred_b": r["pred_b"],
            "correct_b": r["correct_b"],
            "pred_c": r["pred_c"],
            "correct_c": r["correct_c"],
        }
        for b in BETA_VALUES:
            b_key = f"beta_{b:.1f}"
            row[f"pred_soft_{b_key}"] = r["soft_lexicon_betas"][b_key]["pred"]
            row[f"correct_soft_{b_key}"] = r["soft_lexicon_betas"][b_key]["correct"]
        csv_rows.append(row)

    with open(args.csv_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Raw results CSV saved to: {args.csv_output}")


if __name__ == "__main__":
    main()


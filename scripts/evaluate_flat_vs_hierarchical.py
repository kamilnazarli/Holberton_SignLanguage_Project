#!/usr/bin/env python3
"""
Evaluate Flat Static 32-Class Classifier vs. Hierarchical Static Classifier
on the real-pipeline 150-sequence benchmark (75 IV + 75 OOV).

Evaluates across:
  - Scenario 1: Clean real dataset samples
  - Scenario 2: Live camera variation (rotation + jitter)

Uses identical decoder settings:
  - Beam width B = 5
  - Language model weight alpha = 0.6
  - Soft lexicon bonus beta = 3.0
  - 70-word Bigram LM
  - Soft Lexicon mode

Saves results to:
  models/flat_classifier_eval_report.json
  models/flat_classifier_eval_results.csv
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
from scripts.static_model import (
    StaticHierarchicalModel,
    augment_landmarks,
    build_feature_vector_84,
    apply_scaler,
    mlp_forward,
)


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


class FlatStaticModel:
    """Inference wrapper for the exported Flat 32-Class Static MLP JSON model."""

    def __init__(self, model_json_path: str = "models/flat_static_model.json"):
        with open(model_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.scaler = data["scaler"]
        self.model = data["model"]
        self.classes = self.model["classes"]

    def predict_distribution(self, feat84: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Returns top_label, top_conf, and 32-class probability vector."""
        scaled = apply_scaler(feat84, self.scaler)
        candidates = mlp_forward(self.model, scaled)
        dist = dict(candidates)
        top_letter, top_conf = candidates[0]

        prob_vec = np.zeros(len(AZ_ALPHABET), dtype=np.float64)
        floor = 1e-5
        prob_vec[:] = floor
        for letter, p in dist.items():
            if letter in LETTER_TO_IDX:
                prob_vec[LETTER_TO_IDX[letter]] = max(p, floor)
        prob_vec /= np.sum(prob_vec)

        return top_letter, top_conf, prob_vec


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


def build_real_sequences_both_models(
    target_word: str,
    samples_by_letter: Dict[str, List[Dict[str, Any]]],
    model_hier: StaticHierarchicalModel,
    model_flat: FlatStaticModel,
    sample_index: int = 0,
    apply_jitter: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
    prob_seq_hier = []
    prob_seq_flat = []
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

        # 1. Hierarchical distribution
        res_h = model_hier.predict_from_feature_vector(feat84, mode="soft", top_k=2)
        prob_dict_h = res_h.get("distribution", {})
        prob_vec_h = np.zeros(len(AZ_ALPHABET), dtype=np.float64)
        floor = 1e-5
        prob_vec_h[:] = floor
        for letter, p in prob_dict_h.items():
            if letter in LETTER_TO_IDX:
                prob_vec_h[LETTER_TO_IDX[letter]] = max(p, floor)
        prob_vec_h /= np.sum(prob_vec_h)
        prob_seq_hier.append(prob_vec_h)

        # 2. Flat distribution
        _, _, prob_vec_f = model_flat.predict_distribution(feat84)
        prob_seq_flat.append(prob_vec_f)

    return prob_seq_hier, prob_seq_flat, sample_ids


def evaluate_dataset(
    words: List[str],
    is_iv: bool,
    samples_per_word: int,
    samples_by_letter: Dict[str, List[Dict[str, Any]]],
    model_hier: StaticHierarchicalModel,
    model_flat: FlatStaticModel,
    decoder: BeamSearchDecoder,
    apply_jitter: bool = False,
) -> List[Dict[str, Any]]:
    records = []

    for word in words:
        for s_idx in range(samples_per_word):
            prob_seq_h, prob_seq_f, sample_ids = build_real_sequences_both_models(
                word, samples_by_letter, model_hier, model_flat, sample_index=s_idx, apply_jitter=apply_jitter
            )

            # -------------------------------------------------------------
            # Model 1: Hierarchical Classifier + Decoder
            # -------------------------------------------------------------
            # Greedy
            t0 = time.perf_counter()
            pred_h_greedy, _ = decoder.greedy_decode(prob_seq_h)
            lat_h_greedy = (time.perf_counter() - t0) * 1000.0
            dist_h_g = levenshtein_distance(word, pred_h_greedy)
            corr_h_g = (pred_h_greedy == word)
            match_h_g = sum(1 for c1, c2 in zip(word, pred_h_greedy) if c1 == c2) if len(pred_h_greedy) == len(word) else max(0, len(word) - dist_h_g)

            # Beam + Soft Lexicon (B=5, alpha=0.6, beta=3.0)
            t0 = time.perf_counter()
            hyps_h_beam = decoder.decode(prob_seq_h, beam_width=5, use_lm=True, lexicon_mode="soft", lexicon_word_bonus=3.0)
            lat_h_beam = (time.perf_counter() - t0) * 1000.0
            pred_h_beam = hyps_h_beam[0].sequence if hyps_h_beam else ""
            dist_h_b = levenshtein_distance(word, pred_h_beam)
            corr_h_b = (pred_h_beam == word)
            match_h_b = sum(1 for c1, c2 in zip(word, pred_h_beam) if c1 == c2) if len(pred_h_beam) == len(word) else max(0, len(word) - dist_h_b)

            # -------------------------------------------------------------
            # Model 2: Flat 32-Class Classifier + Decoder
            # -------------------------------------------------------------
            # Greedy
            t0 = time.perf_counter()
            pred_f_greedy, _ = decoder.greedy_decode(prob_seq_f)
            lat_f_greedy = (time.perf_counter() - t0) * 1000.0
            dist_f_g = levenshtein_distance(word, pred_f_greedy)
            corr_f_g = (pred_f_greedy == word)
            match_f_g = sum(1 for c1, c2 in zip(word, pred_f_greedy) if c1 == c2) if len(pred_f_greedy) == len(word) else max(0, len(word) - dist_f_g)

            # Beam + Soft Lexicon (B=5, alpha=0.6, beta=3.0)
            t0 = time.perf_counter()
            hyps_f_beam = decoder.decode(prob_seq_f, beam_width=5, use_lm=True, lexicon_mode="soft", lexicon_word_bonus=3.0)
            lat_f_beam = (time.perf_counter() - t0) * 1000.0
            pred_f_beam = hyps_f_beam[0].sequence if hyps_f_beam else ""
            dist_f_b = levenshtein_distance(word, pred_f_beam)
            corr_f_b = (pred_f_beam == word)
            match_f_b = sum(1 for c1, c2 in zip(word, pred_f_beam) if c1 == c2) if len(pred_f_beam) == len(word) else max(0, len(word) - dist_f_b)

            records.append({
                "target_word": word,
                "word_length": len(word),
                "is_in_vocabulary": is_iv,
                "apply_jitter": apply_jitter,
                "sample_index": s_idx,
                "sample_ids": sample_ids,
                # Hierarchical
                "pred_h_greedy": pred_h_greedy,
                "correct_h_greedy": corr_h_g,
                "char_matches_h_g": match_h_g,
                "dist_h_g": dist_h_g,
                "lat_h_g": round(lat_h_greedy, 4),
                "pred_h_beam": pred_h_beam,
                "correct_h_beam": corr_h_b,
                "char_matches_h_b": match_h_b,
                "dist_h_b": dist_h_b,
                "lat_h_b": round(lat_h_beam, 4),
                # Flat
                "pred_f_greedy": pred_f_greedy,
                "correct_f_greedy": corr_f_g,
                "char_matches_f_g": match_f_g,
                "dist_f_g": dist_f_g,
                "lat_f_g": round(lat_f_greedy, 4),
                "pred_f_beam": pred_f_beam,
                "correct_f_beam": corr_f_b,
                "char_matches_f_b": match_f_b,
                "dist_f_b": dist_f_b,
                "lat_f_b": round(lat_f_beam, 4),
                # Rescued / Hurt comparisons between Hierarchical Beam vs Flat Beam
                "flat_rescued_over_hier": (not corr_h_b and corr_f_b),
                "flat_hurt_over_hier": (corr_h_b and not corr_f_b),
            })

    return records


def compute_metrics(records: List[Dict[str, Any]], pred_key: str, corr_key: str, match_key: str, dist_key: str, lat_key: str, base_corr_key: str) -> Dict[str, Any]:
    n_words = len(records)
    total_chars = sum(r["word_length"] for r in records)
    if n_words == 0:
        return {}

    word_acc = sum(1 for r in records if r[corr_key]) / n_words
    char_acc = sum(r[match_key] for r in records) / total_chars
    wer = sum(1 for r in records if not r[corr_key]) / n_words
    cer = sum(r[dist_key] for r in records) / total_chars
    rescued = sum(1 for r in records if (not r[base_corr_key] and r[corr_key]))
    hurt = sum(1 for r in records if (r[base_corr_key] and not r[corr_key]))
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


def analyze_configurations(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "hier_greedy": compute_metrics(records, "pred_h_greedy", "correct_h_greedy", "char_matches_h_g", "dist_h_g", "lat_h_g", "correct_h_greedy"),
        "hier_beam": compute_metrics(records, "pred_h_beam", "correct_h_beam", "char_matches_h_b", "dist_h_b", "lat_h_b", "correct_h_greedy"),
        "flat_greedy": compute_metrics(records, "pred_f_greedy", "correct_f_greedy", "char_matches_f_g", "dist_f_g", "lat_f_g", "correct_h_greedy"),
        "flat_beam": compute_metrics(records, "pred_f_beam", "correct_f_beam", "char_matches_f_b", "dist_f_b", "lat_f_b", "correct_h_beam"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", default="models/.static_landmarks_cache.pkl")
    parser.add_argument("--hier-model-path", default="public/models/azsl_hierarchical_model.json")
    parser.add_argument("--flat-model-path", default="models/flat_static_model.json")
    parser.add_argument("--samples-per-word", type=int, default=5)
    parser.add_argument("--json-output", default="models/flat_classifier_eval_report.json")
    parser.add_argument("--csv-output", default="models/flat_classifier_eval_results.csv")
    args = parser.parse_args()

    print("Loading models...")
    model_hier = StaticHierarchicalModel(args.hier_model_path)
    model_flat = FlatStaticModel(args.flat_model_path)

    print("Initializing decoder (B=5, alpha=0.6, beta=3.0, 70-word Bigram LM)...")
    decoder = BeamSearchDecoder(
        beam_width=5,
        lm_weight=0.6,
        lexicon_word_bonus=3.0,
        lexicon_mode="soft",
        lexicon_words=AZ_LEXICON,
        bigram_corpus=AZ_LEXICON,
    )

    print("Loading real landmark cache...")
    with open(args.cache_path, "rb") as f:
        cache_data = pickle.load(f)

    samples_by_letter: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r, letter in zip(cache_data["records"], cache_data["y_letter"]):
        samples_by_letter[letter].append(r)

    # -------------------------------------------------------------
    # Scenario 1: Clean Real Dataset Samples (N=150)
    # -------------------------------------------------------------
    print("\nEvaluating Scenario 1: Clean Real Dataset Samples (N=150)...")
    clean_iv = evaluate_dataset(IV_WORDS, True, args.samples_per_word, samples_by_letter, model_hier, model_flat, decoder, False)
    clean_oov = evaluate_dataset(OOV_WORDS, False, args.samples_per_word, samples_by_letter, model_hier, model_flat, decoder, False)
    clean_all = clean_iv + clean_oov

    c_iv_m = analyze_configurations(clean_iv)
    c_oov_m = analyze_configurations(clean_oov)
    c_all_m = analyze_configurations(clean_all)

    # -------------------------------------------------------------
    # Scenario 2: Live Camera Variation (N=150)
    # -------------------------------------------------------------
    print("Evaluating Scenario 2: Live Camera Variation (Rotation + Jitter) (N=150)...")
    jitter_iv = evaluate_dataset(IV_WORDS, True, args.samples_per_word, samples_by_letter, model_hier, model_flat, decoder, True)
    jitter_oov = evaluate_dataset(OOV_WORDS, False, args.samples_per_word, samples_by_letter, model_hier, model_flat, decoder, True)
    jitter_all = jitter_iv + jitter_oov

    j_iv_m = analyze_configurations(jitter_iv)
    j_oov_m = analyze_configurations(jitter_oov)
    j_all_m = analyze_configurations(jitter_all)

    # Print Formatted Comparison Tables
    def print_table(title: str, m_iv: Dict[str, Any], m_oov: Dict[str, Any], m_all: Dict[str, Any]):
        print("\n" + "=" * 115)
        print(f"  {title}")
        print("=" * 115)
        print(f"{'Configuration':<38}{'IV Word Acc':<14}{'OOV Word Acc':<15}{'All Word Acc':<15}{'All WER':<10}{'All CER':<10}{'Rescued':<10}{'Hurt'}")
        print("-" * 115)

        cfgs = [
            ("hier_greedy", "1. Hierarchical (Greedy)"),
            ("hier_beam", "2. Hierarchical (Beam + Soft Lex)"),
            ("flat_greedy", "3. Flat 32-Class (Greedy)"),
            ("flat_beam", "4. Flat 32-Class (Beam + Soft Lex)"),
        ]

        for key, name in cfgs:
            iv = m_iv[key]
            oov = m_oov[key]
            comb = m_all[key]
            r_str = str(comb['rescued_count'])
            h_str = str(comb['hurt_count'])
            print(f"{name:<38}{iv['word_acc']:>6.2f}%        {oov['word_acc']:>6.2f}%         {comb['word_acc']:>6.2f}%        {comb['wer']:>5.2f}%    {comb['cer']:>5.2f}%    {r_str:>5s}     {h_str:>5s}")
        print("-" * 115)

    print_table("SCENARIO 1: CLEAN REAL DATASET SAMPLES (N=150: 75 IV + 75 OOV)", c_iv_m, c_oov_m, c_all_m)
    print_table("SCENARIO 2: LIVE CAMERA VARIATION (N=150: 75 IV + 75 OOV)", j_iv_m, j_oov_m, j_all_m)

    # Detailed differences between Hierarchical Beam vs Flat Beam on live variation
    print("\n" + "=" * 115)
    print("  DETAILED HEAD-TO-HEAD: HIERARCHICAL BEAM VS. FLAT BEAM (Live Variation Scenario)")
    print("=" * 115)
    differences = [r for r in jitter_all if r["pred_h_beam"] != r["pred_f_beam"]]
    print(f"Total sequences with different predictions: {len(differences)} / {len(jitter_all)}")

    rescued = [r for r in jitter_all if r["flat_rescued_over_hier"]]
    hurt = [r for r in jitter_all if r["flat_hurt_over_hier"]]

    print(f"  • Flat Model Rescued (Hierarchical wrong -> Flat correct) : {len(rescued)}")
    for r in rescued:
        print(f"    - Target: {r['target_word']:<10} (IV={r['is_in_vocabulary']}) | Hier: {r['pred_h_beam']:<10} -> Flat: {r['pred_f_beam']}")

    print(f"  • Flat Model Hurt (Hierarchical correct -> Flat wrong) : {len(hurt)}")
    for r in hurt:
        print(f"    - Target: {r['target_word']:<10} (IV={r['is_in_vocabulary']}) | Hier: {r['pred_h_beam']:<10} -> Flat: {r['pred_f_beam']}")

    # Save to JSON
    os.makedirs(os.path.dirname(args.json_output), exist_ok=True)
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": {
            "hierarchical_model": args.hier_model_path,
            "flat_model": args.flat_model_path,
            "total_sequences_per_scenario": len(clean_all),
            "iv_sequences_count": len(clean_iv),
            "oov_sequences_count": len(clean_oov),
            "samples_per_word": args.samples_per_word,
            "scenario_1_clean": {
                "iv_metrics": c_iv_m,
                "oov_metrics": c_oov_m,
                "overall_metrics": c_all_m,
            },
            "scenario_2_live_variation": {
                "iv_metrics": j_iv_m,
                "oov_metrics": j_oov_m,
                "overall_metrics": j_all_m,
                "different_predictions_count": len(differences),
                "flat_rescued_count": len(rescued),
                "flat_hurt_count": len(hurt),
            },
        },
    }

    with open(args.json_output, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"\nReport JSON saved to: {args.json_output}")

    # Save to CSV
    os.makedirs(os.path.dirname(args.csv_output), exist_ok=True)
    csv_rows = []
    for r in clean_all + jitter_all:
        csv_rows.append({
            "target_word": r["target_word"],
            "word_length": r["word_length"],
            "is_in_vocabulary": r["is_in_vocabulary"],
            "apply_jitter": r["apply_jitter"],
            "sample_index": r["sample_index"],
            "pred_hier_greedy": r["pred_h_greedy"],
            "correct_hier_greedy": r["correct_h_greedy"],
            "pred_hier_beam": r["pred_h_beam"],
            "correct_hier_beam": r["correct_h_beam"],
            "pred_flat_greedy": r["pred_f_greedy"],
            "correct_flat_greedy": r["correct_f_greedy"],
            "pred_flat_beam": r["pred_f_beam"],
            "correct_flat_beam": r["correct_f_beam"],
            "flat_rescued": r["flat_rescued_over_hier"],
            "flat_hurt": r["flat_hurt_over_hier"],
        })

    with open(args.csv_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Raw results CSV saved to: {args.csv_output}")


if __name__ == "__main__":
    main()


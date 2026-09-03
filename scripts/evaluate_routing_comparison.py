#!/usr/bin/env python3
"""
Comprehensive Benchmarking of Hard Routing vs. Soft Multi-Cluster Routing.

Compares:
  1. HARD ROUTING (K=1, baseline)
  2. SOFT ROUTING (K=2)
  3. SOFT ROUTING (K=3)
  4. SOFT ROUTING (K=all / K=6)

Reports:
  - Accuracy & Macro F1
  - Per-class F1 (especially G, P, Ş)
  - Confusion breakdown
  - Real-world inference latency per prediction
  - Level-2 classifiers evaluated per frame
  - Rescued vs hurt samples analysis
  - 5-fold out-of-fold generalization evaluation
"""

import argparse
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Tuple

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.extract_azsl_model import (
    AZ_ALPHABET,
    CLUSTERS,
    LEVEL2_NARROW_INDICES,
    export_mlp,
    export_scaler,
    make_mlp,
)
from scripts.static_model import StaticHierarchicalModel, apply_scaler, mlp_forward


def benchmark_model_latency(model: StaticHierarchicalModel, feat84: np.ndarray, mode: str, top_k: int, iterations: int = 5000) -> float:
    """Measures average inference latency in milliseconds over multiple runs."""
    # Warmup
    for _ in range(100):
        model.predict_from_feature_vector(feat84, mode=mode, top_k=top_k)
    t0 = time.perf_counter()
    for _ in range(iterations):
        model.predict_from_feature_vector(feat84, mode=mode, top_k=top_k)
    t1 = time.perf_counter()
    return ((t1 - t0) / iterations) * 1000.0  # ms


def evaluate_dataset(
    model: StaticHierarchicalModel,
    X: np.ndarray,
    y_true: np.ndarray,
    mode: str = "hard",
    top_k: int = 1,
) -> Dict[str, Any]:
    preds = []
    confs = []
    eval_counts = []

    for i in range(len(X)):
        res = model.predict_from_feature_vector(X[i], mode=mode, top_k=top_k)
        preds.append(res["label"])
        confs.append(res["confidence"])
        eval_counts.append(res["evaluatedClusters"])

    preds = np.array(preds)
    acc = float(np.mean(preds == y_true))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))

    classes = sorted(list(set(y_true)))
    p, r, f, s = precision_recall_fscore_support(y_true, preds, labels=classes, zero_division=0)
    per_class = {}
    for idx, c in enumerate(classes):
        per_class[c] = {
            "precision": float(p[idx]),
            "recall": float(r[idx]),
            "f1": float(f[idx]),
            "support": int(s[idx]),
        }

    cm = confusion_matrix(y_true, preds, labels=classes)

    return {
        "mode": mode,
        "top_k": top_k,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "avg_evaluated_clusters": float(np.mean(eval_counts)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "predictions": preds.tolist(),
        "classes": classes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-path", default="models/.static_landmarks_cache.pkl")
    parser.add_argument("--model-path", default="public/models/azsl_hierarchical_model.json")
    parser.add_argument("--output", default="models/routing_comparison_report.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isfile(args.cache_path):
        sys.exit(f"Cache file not found: {args.cache_path}")
    if not os.path.isfile(args.model_path):
        sys.exit(f"Model file not found: {args.model_path}")

    print("Loading pre-trained hierarchical static model...")
    model = StaticHierarchicalModel(args.model_path)

    print("Loading landmark records from cache...")
    with open(args.cache_path, "rb") as f:
        cache_data = pickle.load(f)

    records = cache_data["records"]
    y_true = np.array(cache_data["y_letter"])
    X = np.array([r["feat84"] for r in records], dtype=np.float64)
    print(f"Total benchmark samples: {len(X)} across {len(set(y_true))} classes\n")

    configs = [
        ("Hard (K=1)", "hard", 1),
        ("Soft (K=2)", "soft", 2),
        ("Soft (K=3)", "soft", 3),
        ("Soft (K=all)", "soft", 6),
    ]

    results = {}
    latencies = {}

    sample_feat = X[0]
    print("Measuring inference latencies (5,000 runs each)...")
    for name, mode, top_k in configs:
        lat = benchmark_model_latency(model, sample_feat, mode=mode, top_k=top_k, iterations=5000)
        latencies[name] = lat
        print(f"  {name:<16}: {lat:.4f} ms per prediction")

    print("\nRunning full-dataset evaluation across all configurations...")
    for name, mode, top_k in configs:
        t0 = time.time()
        res = evaluate_dataset(model, X, y_true, mode=mode, top_k=top_k)
        res["latency_ms"] = latencies[name]
        results[name] = res
        print(f"  {name:<16}: Acc = {res['accuracy']*100:.2f}% | Macro F1 = {res['macro_f1']:.4f} | L2 Eval = {res['avg_evaluated_clusters']:.1f} ({time.time()-t0:.2f}s)")

    # 1. Comparative Summary Table
    print("\n" + "=" * 80)
    print("  STATIC DISPATCHER COMPARISON: HARD VS. SOFT MULTI-CLUSTER ROUTING")
    print("=" * 80)
    print(f"{'Configuration':<16}{'Accuracy':<12}{'Macro F1':<12}{'L2 Evaluated':<15}{'Latency (ms)':<14}{'Delta Acc':<10}")
    print("-" * 80)
    base_acc = results["Hard (K=1)"]["accuracy"]
    base_f1 = results["Hard (K=1)"]["macro_f1"]

    for name, _, _ in configs:
        r = results[name]
        d_acc = (r["accuracy"] - base_acc) * 100
        print(f"{name:<16}{r['accuracy']*100:>6.2f}%      {r['macro_f1']:>7.4f}     {r['avg_evaluated_clusters']:>5.1f} / 6       {r['latency_ms']:>6.4f} ms       {d_acc:>+5.2f}%")
    print("-" * 80)

    # 2. Focused inspection on G, P, and Ş
    target_letters = ["G", "P", "Ş"]
    print("\n" + "=" * 80)
    print("  FOCUSED INSPECTION: G, P, and Ş")
    print("=" * 80)
    print(f"{'Letter':<8}{'Metric':<12}{'Hard (K=1)':<14}{'Soft (K=2)':<14}{'Soft (K=all)':<14}")
    print("-" * 80)
    for letter in target_letters:
        for metric in ["precision", "recall", "f1"]:
            h_val = results["Hard (K=1)"]["per_class"].get(letter, {}).get(metric, 0.0)
            s2_val = results["Soft (K=2)"]["per_class"].get(letter, {}).get(metric, 0.0)
            sall_val = results["Soft (K=all)"]["per_class"].get(letter, {}).get(metric, 0.0)
            label_str = letter if metric == "precision" else ""
            print(f"{label_str:<8}{metric.capitalize():<12}{h_val:>6.2f}        {s2_val:>6.2f}        {sall_val:>6.2f}")
        supp = results["Hard (K=1)"]["per_class"].get(letter, {}).get("support", 0)
        print(f"{'':<8}{'Support':<12}{supp:>6d}        {supp:>6d}        {supp:>6d}")
        print("-" * 55)

    # 3. 5-Fold Cross-Validation Analysis (Out-of-Fold Generalization)
    print("\n" + "=" * 80)
    print("  5-FOLD CROSS-VALIDATION (OUT-OF-FOLD GENERALIZATION)")
    print("=" * 80)
    LETTER_TO_CLUSTER = {}
    for cid, letters in CLUSTERS.items():
        for l in letters:
            LETTER_TO_CLUSTER[l] = int(cid)

    y_cluster = np.array([LETTER_TO_CLUSTER[l] for l in y_true])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    cv_results = {}
    for k_val in [1, 2, 3, 6]:
        fold_accs = []
        for train_idx, test_idx in skf.split(X, y_true):
            scaler80 = StandardScaler().fit(X[train_idx])
            l1 = make_mlp((48, 24), 1e-3, args.seed)
            l1.fit(scaler80.transform(X[train_idx]), y_cluster[train_idx])

            l2_dict = {}
            for cid, letters in CLUSTERS.items():
                mask = np.isin(y_true[train_idx], letters)
                if cid == 6:
                    sub_sc = StandardScaler().fit(X[train_idx][mask])
                    clf = make_mlp((64, 32), 1e-3, args.seed)
                    clf.fit(sub_sc.transform(X[train_idx][mask]), y_true[train_idx][mask])
                else:
                    sub_X = X[train_idx][:, LEVEL2_NARROW_INDICES][mask]
                    sub_sc = StandardScaler().fit(sub_X)
                    clf = make_mlp((16,), 1e-2, args.seed)
                    clf.fit(sub_sc.transform(sub_X), y_true[train_idx][mask])
                l2_dict[cid] = (export_mlp(clf), export_scaler(sub_sc))

            l1_export = export_mlp(l1)
            scaler80_export = export_scaler(scaler80)

            correct = 0
            for i in test_idx:
                feat = X[i]
                l1_sc = apply_scaler(feat, scaler80_export)
                c_cands = mlp_forward(l1_export, l1_sc)

                eval_clusters = c_cands[:k_val]
                sum_c = sum(c[1] for c in eval_clusters)
                if sum_c < 1e-9: sum_c = 1e-9

                l_probs = {}
                for cid, c_prob in eval_clusters:
                    clf_e, sc_e = l2_dict[cid]
                    sub_feat = feat if cid == 6 else feat[LEVEL2_NARROW_INDICES]
                    sub_sc = apply_scaler(sub_feat, sc_e)
                    sub_cands = mlp_forward(clf_e, sub_sc)
                    for l_name, l_prob in sub_cands:
                        l_probs[l_name] = (c_prob * l_prob) / sum_c

                pred = max(l_probs.items(), key=lambda x: x[1])[0]
                if pred == y_true[i]:
                    correct += 1
            fold_accs.append(correct / len(test_idx))

        k_name = f"K={k_val}" if k_val < 6 else "K=all"
        cv_results[k_name] = {
            "mean": float(np.mean(fold_accs)),
            "per_fold": [round(a, 4) for a in fold_accs],
        }
        print(f"  {k_name:<10}: 5-Fold Mean = {np.mean(fold_accs)*100:.2f}% (per-fold: {[round(a*100, 2) for a in fold_accs]})")

    # 4. Identification of Rescued vs. Hurt Examples
    print("\n" + "=" * 80)
    print("  IDENTIFICATION OF RESCUED AND HURT EXAMPLES")
    print("=" * 80)
    # Check on fold 2 where difference occurred
    train_idx, test_idx = list(skf.split(X, y_true))[1]
    scaler80 = StandardScaler().fit(X[train_idx])
    l1 = make_mlp((48, 24), 1e-3, args.seed)
    l1.fit(scaler80.transform(X[train_idx]), y_cluster[train_idx])
    l2_dict = {}
    for cid, letters in CLUSTERS.items():
        mask = np.isin(y_true[train_idx], letters)
        if cid == 6:
            sub_sc = StandardScaler().fit(X[train_idx][mask])
            clf = make_mlp((64, 32), 1e-3, args.seed)
            clf.fit(sub_sc.transform(X[train_idx][mask]), y_true[train_idx][mask])
        else:
            sub_X = X[train_idx][:, LEVEL2_NARROW_INDICES][mask]
            sub_sc = StandardScaler().fit(sub_X)
            clf = make_mlp((16,), 1e-2, args.seed)
            clf.fit(sub_sc.transform(sub_X), y_true[train_idx][mask])
        l2_dict[cid] = (export_mlp(clf), export_scaler(sub_sc))
    l1_export = export_mlp(l1)
    scaler80_export = export_scaler(scaler80)

    rescued_cases = []
    hurt_cases = []
    for i in test_idx:
        feat = X[i]
        true_l = y_true[i]
        true_c = LETTER_TO_CLUSTER[true_l]

        l1_sc = apply_scaler(feat, scaler80_export)
        c_cands = mlp_forward(l1_export, l1_sc)

        # Hard routing (K=1)
        top_c = c_cands[0][0]
        clf_h, sc_h = l2_dict[top_c]
        feat_h = feat if top_c == 6 else feat[LEVEL2_NARROW_INDICES]
        l_c_h = mlp_forward(clf_h, apply_scaler(feat_h, sc_h))
        pred_hard = l_c_h[0][0]

        # Soft routing (K=2)
        eval_c = c_cands[:2]
        sum_c = sum(c[1] for c in eval_c)
        l_probs_s = {}
        for cid, c_prob in eval_c:
            clf_s, sc_s = l2_dict[cid]
            feat_s = feat if cid == 6 else feat[LEVEL2_NARROW_INDICES]
            l_c_s = mlp_forward(clf_s, apply_scaler(feat_s, sc_s))
            for l_n, l_p in l_c_s:
                l_probs_s[l_n] = (c_prob * l_p) / sum_c
        pred_soft = max(l_probs_s.items(), key=lambda x: x[1])[0]

        if pred_hard != true_l and pred_soft == true_l:
            rescued_cases.append({
                "sample_idx": int(i),
                "true_label": true_l,
                "true_cluster": int(true_c),
                "hard_pred": pred_hard,
                "hard_cluster": int(top_c),
                "l1_cands": [(int(c[0]), round(float(c[1]), 4)) for c in c_cands[:2]],
                "soft_top3": [(k, round(v, 4)) for k, v in sorted(l_probs_s.items(), key=lambda x: -x[1])[:3]],
            })
        elif pred_hard == true_l and pred_soft != true_l:
            hurt_cases.append({
                "sample_idx": int(i),
                "true_label": true_l,
                "hard_pred": pred_hard,
                "soft_pred": pred_soft,
            })

    print(f"Total Rescued Cases (Hard=Wrong, Soft=Correct): {len(rescued_cases)}")
    for rc in rescued_cases:
        print(f"  • Sample #{rc['sample_idx']}: Ground Truth '{rc['true_label']}' (in Cluster {rc['true_cluster']})")
        print(f"    - Hard Routing: Level-1 selected Cluster {rc['hard_cluster']} ({rc['l1_cands'][0][1]*100:.1f}%), forced prediction '{rc['hard_pred']}' (INCORRECT)")
        print(f"    - Soft Routing: Level-1 evaluated Cluster {rc['l1_cands'][0][0]} and Cluster {rc['l1_cands'][1][0]}, joint probability resolved to '{rc['true_label']}' ({rc['soft_top3'][0][1]*100:.1f}%) (CORRECT!)")

    print(f"Total Hurt Cases (Hard=Correct, Soft=Wrong): {len(hurt_cases)}")
    if not hurt_cases:
        print("  • None! Zero samples were degraded by enabling soft routing.")

    # Save complete report
    final_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_size": len(X),
        "configurations": results,
        "cross_validation_5fold": cv_results,
        "rescued_cases": rescued_cases,
        "hurt_cases": hurt_cases,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    print(f"\nFull report saved to: {args.output}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Train and evaluate a single Flat 32-Class Static MLP Classifier against the
Two-Level Hierarchical Clustered Classifier on Azerbaijani Sign Language (AzSLD).

Uses the exact same 84-D geometric feature vectors and 5-fold stratified cross-validation.
Evaluates:
  - Overall accuracy, Macro F1, Per-class accuracy
  - Confusion matrix
  - Specific confusion pairs (H <-> P, G <-> S, G <-> Ş, etc.)
Exports the trained flat model to models/flat_static_model.json and .pkl.
"""

import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.extract_azsl_model import (
    AZ_ALPHABET,
    CLUSTERS,
    LETTER_TO_CLUSTER,
    LEVEL2_NARROW_INDICES,
    FULL_VECTOR_LENGTH,
    export_mlp,
    export_scaler,
    make_mlp,
)


def mlp_forward_probs(clf: MLPClassifier, X: np.ndarray) -> np.ndarray:
    """Computes softmax output probabilities for an MLPClassifier."""
    return clf.predict_proba(X)


def run_hierarchical_predict(
    level1_clf: MLPClassifier,
    level1_scaler: StandardScaler,
    level2_models: Dict[int, Tuple[MLPClassifier, StandardScaler]],
    X_test: np.ndarray,
    mode: str = "soft",
    top_k: int = 2,
) -> Tuple[List[str], List[Dict[str, float]]]:
    """Runs hierarchical prediction on a test batch."""
    l1_scaled = level1_scaler.transform(X_test)
    l1_probs = level1_clf.predict_proba(l1_scaled)  # shape (N, 6)
    l1_classes = level1_clf.classes_  # cluster ids [1, 2, 3, 4, 5, 6]

    preds = []
    dist_list = []

    for i in range(len(X_test)):
        row_feat = X_test[i:i + 1]
        probs_row = l1_probs[i]
        sorted_indices = np.argsort(-probs_row)

        if mode == "hard" or top_k == 1:
            top_cid = int(l1_classes[sorted_indices[0]])
            clf, sub_scaler = level2_models[top_cid]
            if top_cid == 6:
                feat = sub_scaler.transform(row_feat)
            else:
                feat = sub_scaler.transform(row_feat[:, LEVEL2_NARROW_INDICES])
            p_letter = clf.predict_proba(feat)[0]
            letter_dist = {cls: float(p) for cls, p in zip(clf.classes_, p_letter)}
            top_letter = clf.classes_[np.argmax(p_letter)]
            preds.append(top_letter)
            dist_list.append(letter_dist)
        else:
            k = min(top_k, len(sorted_indices))
            eval_clusters = [(int(l1_classes[idx]), probs_row[idx]) for idx in sorted_indices[:k]]
            sum_prob = sum(p for _, p in eval_clusters) or 1e-9

            letter_probs: Dict[str, float] = {}
            for cid, c_prob in eval_clusters:
                clf, sub_scaler = level2_models[cid]
                if cid == 6:
                    feat = sub_scaler.transform(row_feat)
                else:
                    feat = sub_scaler.transform(row_feat[:, LEVEL2_NARROW_INDICES])
                p_letter = clf.predict_proba(feat)[0]
                for cls, p in zip(clf.classes_, p_letter):
                    joint = (c_prob * p) / sum_prob
                    letter_probs[cls] = letter_probs.get(cls, 0.0) + float(joint)

            top_letter = max(letter_probs.items(), key=lambda item: item[1])[0]
            preds.append(top_letter)
            dist_list.append(letter_probs)

    return preds, dist_list


def main():
    cache_path = "models/.static_landmarks_cache.pkl"
    if not os.path.exists(cache_path):
        sys.exit(f"Cache file {cache_path} not found.")

    print(f"Loading cached landmark data from {cache_path}...")
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)

    records = cache_data["records"]
    y_letter = cache_data["y_letter"]
    X = np.array([r["feat84"] for r in records], dtype=np.float64)
    y_cluster = np.array([LETTER_TO_CLUSTER[l] for l in y_letter])

    print(f"Dataset shape: X={X.shape}, y={len(y_letter)} samples across {len(np.unique(y_letter))} classes.")

    seed = 42
    folds = 5
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    # Cross-validation tracking
    # 1. Hierarchical Hard (K=1)
    hier_hard_preds = np.empty_like(y_letter)
    # 2. Hierarchical Soft (K=2)
    hier_soft_preds = np.empty_like(y_letter)
    # 3. Flat 32-Class MLP
    flat_preds = np.empty_like(y_letter)
    flat_probs = np.zeros((len(y_letter), len(AZ_ALPHABET)), dtype=np.float64)

    hier_hard_fold_accs = []
    hier_soft_fold_accs = []
    flat_fold_accs = []

    print("\nRunning 5-Fold Stratified Cross-Validation...")
    print("-" * 75)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_letter)):
        t0 = time.time()
        X_train, y_train_letter, y_train_cluster = X[train_idx], y_letter[train_idx], y_cluster[train_idx]
        X_test, y_test_letter = X[test_idx], y_letter[test_idx]

        # ------------------------------------------------------------------
        # Train Hierarchical Model for this fold
        # ------------------------------------------------------------------
        l1_scaler = StandardScaler().fit(X_train)
        level1_clf = make_mlp((48, 24), 1e-3, seed + fold_idx)
        level1_clf.fit(l1_scaler.transform(X_train), y_train_cluster)

        level2_models = {}
        for cid, letters in CLUSTERS.items():
            mask = np.isin(y_train_letter, letters)
            if cid == 6:
                sub_scaler = StandardScaler().fit(X_train[mask])
                clf = make_mlp((64, 32), 1e-3, seed + fold_idx)
                clf.fit(sub_scaler.transform(X_train[mask]), y_train_letter[mask])
            else:
                sub_X = X_train[mask][:, LEVEL2_NARROW_INDICES]
                sub_scaler = StandardScaler().fit(sub_X)
                clf = make_mlp((16,), 1e-2, seed + fold_idx)
                clf.fit(sub_scaler.transform(sub_X), y_train_letter[mask])
            level2_models[cid] = (clf, sub_scaler)

        # Predict Hierarchical Hard
        preds_hh, _ = run_hierarchical_predict(level1_clf, l1_scaler, level2_models, X_test, mode="hard", top_k=1)
        hier_hard_preds[test_idx] = preds_hh
        acc_hh = np.mean(preds_hh == y_test_letter)
        hier_hard_fold_accs.append(acc_hh)

        # Predict Hierarchical Soft
        preds_hs, _ = run_hierarchical_predict(level1_clf, l1_scaler, level2_models, X_test, mode="soft", top_k=2)
        hier_soft_preds[test_idx] = preds_hs
        acc_hs = np.mean(preds_hs == y_test_letter)
        hier_soft_fold_accs.append(acc_hs)

        # ------------------------------------------------------------------
        # Train Flat 32-Class MLP for this fold
        # Architecture: 84 -> 128 (ReLU) -> 64 (ReLU) -> 32 (Softmax)
        # ------------------------------------------------------------------
        flat_scaler = StandardScaler().fit(X_train)
        flat_clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            max_iter=3000,
            early_stopping=False,
            n_iter_no_change=25,
            random_state=seed + fold_idx,
        )
        flat_clf.fit(flat_scaler.transform(X_train), y_train_letter)

        preds_flat = flat_clf.predict(flat_scaler.transform(X_test))
        probs_flat = flat_clf.predict_proba(flat_scaler.transform(X_test))
        flat_preds[test_idx] = preds_flat
        flat_probs[test_idx] = probs_flat

        acc_flat = np.mean(preds_flat == y_test_letter)
        flat_fold_accs.append(acc_flat)

        elapsed = time.time() - t0
        print(f"Fold {fold_idx + 1}/{folds} ({elapsed:.1f}s) | Hier Hard: {acc_hh * 100:.2f}% | Hier Soft (K=2): {acc_hs * 100:.2f}% | Flat 32-Class: {acc_flat * 100:.2f}%")

    print("-" * 75)

    # Summary Metrics
    mean_hh = np.mean(hier_hard_fold_accs) * 100
    mean_hs = np.mean(hier_soft_fold_accs) * 100
    mean_flat = np.mean(flat_fold_accs) * 100

    f1_hh = f1_score(y_letter, hier_hard_preds, average="macro") * 100
    f1_hs = f1_score(y_letter, hier_soft_preds, average="macro") * 100
    f1_flat = f1_score(y_letter, flat_preds, average="macro") * 100

    print("\n" + "=" * 75)
    print("  5-FOLD CROSS-VALIDATION SUMMARY (ISOLATED LETTERS)")
    print("=" * 75)
    print(f"{'Model Architecture':<38}{'Mean Accuracy':<18}{'Macro F1':<12}")
    print("-" * 75)
    print(f"{'Hierarchical (Hard Routing K=1)':<38}{mean_hh:>6.2f}%            {f1_hh:>6.2f}%")
    print(f"{'Hierarchical (Soft Routing K=2)':<38}{mean_hs:>6.2f}%            {f1_hs:>6.2f}%")
    print(f"{'Flat 32-Class MLP (128, 64)':<38}{mean_flat:>6.2f}%            {f1_flat:>6.2f}%")
    print("-" * 75)
    diff = mean_flat - mean_hs
    print(f"Flat vs. Hierarchical Soft Improvement: {diff:+.2f} percentage points (F1: {f1_flat - f1_hs:+.2f} pp)")

    # Per-Class Accuracy Analysis
    print("\n" + "=" * 75)
    print("  PER-CLASS ACCURACY COMPARISON")
    print("=" * 75)
    print(f"{'Class':<8}{'Samples':<10}{'Hier Hard':<14}{'Hier Soft (K=2)':<18}{'Flat 32-Class':<14}{'Flat Diff'}")
    print("-" * 75)

    per_class_results = {}
    classes = sorted(list(np.unique(y_letter)))
    for c in classes:
        mask = (y_letter == c)
        total = np.sum(mask)
        acc_c_hh = np.mean(hier_hard_preds[mask] == c) * 100
        acc_c_hs = np.mean(hier_soft_preds[mask] == c) * 100
        acc_c_flat = np.mean(flat_preds[mask] == c) * 100
        d_c = acc_c_flat - acc_c_hs
        sign = "+" if d_c >= 0 else ""
        print(f"{c:<8}{total:<10}{acc_c_hh:>6.2f}%        {acc_c_hs:>6.2f}%            {acc_c_flat:>6.2f}%         {sign}{d_c:>.2f}%")
        per_class_results[c] = {
            "samples": int(total),
            "hier_hard_acc": round(acc_c_hh, 2),
            "hier_soft_acc": round(acc_c_hs, 2),
            "flat_acc": round(acc_c_flat, 2),
            "diff_vs_hier_soft": round(d_c, 2),
        }

    # Confusion Matrix & Critical Pairs
    cm_hs = confusion_matrix(y_letter, hier_soft_preds, labels=AZ_ALPHABET)
    cm_flat = confusion_matrix(y_letter, flat_preds, labels=AZ_ALPHABET)

    print("\n" + "=" * 75)
    print("  CRITICAL CONFUSION PAIR INSPECTION")
    print("=" * 75)
    critical_pairs = [
        ("H", "P"),
        ("G", "S"),
        ("G", "Ş"),
        ("B", "R"),
        ("C", "K"),
        ("A", "Ə"),
        ("J", "Ç"),
        ("J", "F"),
    ]

    print(f"{'Confusion Pair':<18}{'Hierarchical Soft Errors':<28}{'Flat 32-Class Errors':<22}{'Error Reduction'}")
    print("-" * 75)
    pair_results = {}
    for c1, c2 in critical_pairs:
        idx1 = AZ_ALPHABET.index(c1)
        idx2 = AZ_ALPHABET.index(c2)

        # Errors: true c1 predicted as c2, and true c2 predicted as c1
        err_hs = cm_hs[idx1, idx2] + cm_hs[idx2, idx1]
        err_flat = cm_flat[idx1, idx2] + cm_flat[idx2, idx1]
        reduction = err_hs - err_flat
        red_pct = f"{reduction:+d} errors" if reduction != 0 else "0 (no change)"

        pair_name = f"{c1} <-> {c2}"
        print(f"{pair_name:<18}{err_hs:<28}{err_flat:<22}{red_pct}")
        pair_results[pair_name] = {
            "hier_soft_errors": int(err_hs),
            "flat_errors": int(err_flat),
            "reduction": int(reduction),
        }

    # ------------------------------------------------------------------
    # Train Final Flat 32-Class Classifier on full dataset & export
    # ------------------------------------------------------------------
    print("\nTraining final Flat 32-Class Classifier on all 2,892 samples for export...")
    final_scaler = StandardScaler().fit(X)
    final_clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        max_iter=3000,
        early_stopping=False,
        n_iter_no_change=25,
        random_state=seed,
    )
    final_clf.fit(final_scaler.transform(X), y_letter)
    train_score = final_clf.score(final_scaler.transform(X), y_letter) * 100
    print(f"Final Flat Classifier fit complete. Full dataset training accuracy: {train_score:.2f}%")

    # Export to JSON
    json_export = {
        "format": "azsl_flat_static_model_v1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "architecture": {
            "type": "flat_mlp",
            "input_dimension": FULL_VECTOR_LENGTH,
            "hidden_layers": [128, 64],
            "output_classes": len(AZ_ALPHABET),
            "hidden_activation": "relu",
            "output_activation": "softmax",
        },
        "scaler": export_scaler(final_scaler),
        "model": export_mlp(final_clf),
        "cv_summary": {
            "folds": folds,
            "mean_accuracy": round(mean_flat, 2),
            "macro_f1": round(f1_flat, 2),
            "per_fold": [round(a * 100, 2) for a in flat_fold_accs],
        },
    }

    json_path = "models/flat_static_model.json"
    pkl_path = "models/flat_static_model.pkl"
    report_path = "models/flat_model_cv_report.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2, ensure_ascii=False)
    print(f"Exported JSON flat model: {json_path}")

    with open(pkl_path, "wb") as f:
        pickle.dump({"scaler": final_scaler, "classifier": final_clf, "classes": final_clf.classes_}, f)
    print(f"Exported PKL flat model: {pkl_path}")

    cv_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hierarchical_hard": {
            "mean_accuracy": round(mean_hh, 2),
            "macro_f1": round(f1_hh, 2),
            "per_fold": [round(a * 100, 2) for a in hier_hard_fold_accs],
        },
        "hierarchical_soft": {
            "mean_accuracy": round(mean_hs, 2),
            "macro_f1": round(f1_hs, 2),
            "per_fold": [round(a * 100, 2) for a in hier_soft_fold_accs],
        },
        "flat_32_class": {
            "mean_accuracy": round(mean_flat, 2),
            "macro_f1": round(f1_flat, 2),
            "per_fold": [round(a * 100, 2) for a in flat_fold_accs],
        },
        "per_class": per_class_results,
        "critical_confusion_pairs": pair_results,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(cv_report, f, indent=2, ensure_ascii=False)
    print(f"Saved CV evaluation report: {report_path}")


if __name__ == "__main__":
    main()


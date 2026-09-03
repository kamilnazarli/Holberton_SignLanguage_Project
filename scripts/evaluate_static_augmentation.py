#!/usr/bin/env python3
"""
Comparative Evaluation of Baseline vs Augmented Static AzSL Models.

Trains and evaluates:
  Configuration A: BASELINE (unaugmented training)
  Configuration B: AUGMENTED (landmark augmentation: rotation, scale, translation, jitter)
Under identical architectures, random seed, and unaugmented test sets.
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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import types

# Ensure doc_controls is mocked if tensorflow docs are missing / on Python 3.13
if "tensorflow" not in sys.modules:
    tf = types.ModuleType("tensorflow")
    tf_tools = types.ModuleType("tensorflow.tools")
    tf_docs = types.ModuleType("tensorflow.tools.docs")

    class _DocControls:
        @staticmethod
        def do_not_doc_inheritable(obj): return obj
        @staticmethod
        def do_not_generate_docs(obj): return obj

    tf_docs.doc_controls = _DocControls
    tf.tools = tf_tools
    tf_tools.docs = tf_docs
    sys.modules["tensorflow"] = tf
    sys.modules["tensorflow.tools"] = tf_tools
    sys.modules["tensorflow.tools.docs"] = tf_docs

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from scripts.extract_azsl_model import (
    AZ_ALPHABET,
    CLUSTERS,
    FULL_VECTOR_LENGTH,
    LETTER_TO_CLUSTER,
    LEVEL2_NARROW_INDICES,
    build_dataset,
    end_to_end_cross_validate,
    extract_all,
    make_mlp,
)


def train_hierarchical_model(
    X_train: np.ndarray,
    y_train_letter: np.ndarray,
    y_train_cluster: np.ndarray,
    seed: int = 42,
) -> Tuple[StandardScaler, MLPClassifier, Dict[int, Tuple[MLPClassifier, StandardScaler]]]:
    """Fits Level 1 dispatcher and Level 2 cluster models."""
    scaler80 = StandardScaler().fit(X_train)
    level1 = make_mlp((48, 24), 1e-3, seed)
    level1.fit(scaler80.transform(X_train), y_train_cluster)

    level2_models = {}
    for cid, letters in CLUSTERS.items():
        mask = np.isin(y_train_letter, letters)
        if cid == 6:
            sub_scaler = StandardScaler().fit(X_train[mask])
            clf = make_mlp((64, 32), 1e-3, seed)
            clf.fit(sub_scaler.transform(X_train[mask]), y_train_letter[mask])
        else:
            sub_X = X_train[mask][:, LEVEL2_NARROW_INDICES]
            sub_scaler = StandardScaler().fit(sub_X)
            clf = make_mlp((16,), 1e-2, seed)
            clf.fit(sub_scaler.transform(sub_X), y_train_letter[mask])
        level2_models[cid] = (clf, sub_scaler)

    return scaler80, level1, level2_models


def predict_hierarchical(
    X: np.ndarray,
    scaler80: StandardScaler,
    level1: MLPClassifier,
    level2_models: Dict[int, Tuple[MLPClassifier, StandardScaler]],
) -> np.ndarray:
    """Predicts letter labels using two-level hierarchical inference."""
    pred_clusters = level1.predict(scaler80.transform(X))
    final_preds = []
    for i, cid in enumerate(pred_clusters):
        clf, sub_scaler = level2_models[cid]
        if cid == 6:
            feat = scaler80.transform(X[i : i + 1])
        else:
            feat = sub_scaler.transform(X[i : i + 1][:, LEVEL2_NARROW_INDICES])
        final_preds.append(clf.predict(feat)[0])
    return np.array(final_preds)


def evaluate_configuration(
    name: str,
    train_records: List[Dict[str, Any]],
    train_y_letter: np.ndarray,
    test_records: List[Dict[str, Any]],
    test_y_letter: np.ndarray,
    all_records: List[Dict[str, Any]],
    all_y_letter: np.ndarray,
    all_y_cluster: np.ndarray,
    augment: bool,
    aug_copies: int,
    aug_params: Dict[str, Any],
    seed: int = 42,
    folds: int = 5,
) -> Dict[str, Any]:
    print(f"\n{'=' * 65}")
    print(f"  EVALUATING CONFIGURATION: {name.upper()}")
    print(f"{'=' * 65}")

    t0 = time.time()
    rng_train = np.random.RandomState(seed)

    # 1. Build training set
    X_train, y_train_letter, y_train_cluster = build_dataset(
        train_records,
        train_y_letter,
        augment=augment,
        aug_copies=aug_copies,
        aug_params=aug_params,
        rng=rng_train,
    )
    print(f"Training dataset size: {len(X_train)} samples ({'Augmented' if augment else 'Baseline'})")

    # 2. Build test set (STRICTLY UNAUGMENTED)
    X_test = np.array([r["feat84"] for r in test_records], dtype=np.float64)
    y_test_letter = test_y_letter
    print(f"Test dataset size:     {len(X_test)} samples (Strictly Unaugmented)")

    # 3. Train models
    scaler80, level1, level2_models = train_hierarchical_model(
        X_train, y_train_letter, y_train_cluster, seed=seed
    )

    # 4. Predict on unaugmented training samples to measure fit
    X_train_orig = np.array([r["feat84"] for r in train_records], dtype=np.float64)
    train_preds = predict_hierarchical(X_train_orig, scaler80, level1, level2_models)
    train_acc = float(np.mean(train_preds == train_y_letter))
    train_f1 = float(f1_score(train_y_letter, train_preds, average="macro", zero_division=0))

    # 5. Predict on holdout test set
    test_preds = predict_hierarchical(X_test, scaler80, level1, level2_models)
    test_acc = float(np.mean(test_preds == y_test_letter))
    test_f1 = float(f1_score(y_test_letter, test_preds, average="macro", zero_division=0))

    # 6. Run 5-fold cross-validation
    print(f"\nRunning {folds}-fold end-to-end cross-validation...")
    cv_report = end_to_end_cross_validate(
        all_records,
        all_y_letter,
        all_y_cluster,
        seed=seed,
        folds=folds,
        augment=augment,
        aug_copies=aug_copies,
        aug_params=aug_params,
    )
    cv_mean = cv_report["mean"]
    print(f"  {folds}-fold E2E CV accuracy: {cv_mean * 100:.2f}% (per-fold: {cv_report['perFold']})")

    # 7. Detailed metrics & confusion matrix
    classes = sorted(list(set(all_y_letter)))
    cm = confusion_matrix(y_test_letter, test_preds, labels=classes)
    p, r, f, s = precision_recall_fscore_support(
        y_test_letter, test_preds, labels=classes, zero_division=0
    )
    per_class = {}
    for idx, c in enumerate(classes):
        per_class[c] = {
            "precision": float(p[idx]),
            "recall": float(r[idx]),
            "f1": float(f[idx]),
            "support": int(s[idx]),
        }

    elapsed = time.time() - t0
    print(f"Train Acc: {train_acc * 100:.2f}% | Train Macro F1: {train_f1:.4f}")
    print(f"Test  Acc: {test_acc * 100:.2f}% | Test  Macro F1: {test_f1:.4f}")
    print(f"Completed in {elapsed:.1f}s")

    return {
        "name": name,
        "augment": augment,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_acc": train_acc,
        "train_f1": train_f1,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "cv_mean": cv_mean,
        "cv_folds": cv_report["perFold"],
        "per_class": per_class,
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "predictions": test_preds.tolist(),
        "ground_truth": y_test_letter.tolist(),
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/AzSLD_Fingerspelling")
    parser.add_argument("--model-path", default="public/models/hand_landmarker.task")
    parser.add_argument("--output", default="models/static_augmentation_eval_report.json")
    parser.add_argument("--max-per-class", type=int, default=250)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--folds", type=int, default=5)
    # Augmentation hyperparameters
    parser.add_argument("--aug-copies", type=int, default=1)
    parser.add_argument("--max-rot-x", type=float, default=8.0)
    parser.add_argument("--max-rot-y", type=float, default=8.0)
    parser.add_argument("--max-rot-z", type=float, default=10.0)
    parser.add_argument("--scale-min", type=float, default=0.92)
    parser.add_argument("--scale-max", type=float, default=1.08)
    parser.add_argument("--max-trans", type=float, default=0.02)
    parser.add_argument("--jitter-std", type=float, default=0.008)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    aug_params = {
        "max_angles": (args.max_rot_x, args.max_rot_y, args.max_rot_z),
        "scale_range": (args.scale_min, args.scale_max),
        "max_trans": args.max_trans,
        "jitter_std": args.jitter_std,
    }

    print("Loading AzSL landmark dataset...")
    records, y_letter, summary = extract_all(
        args.data_dir,
        args.model_path,
        args.min_confidence,
        args.max_per_class,
        args.seed,
        no_cache=args.no_cache,
    )
    y_cluster = np.array([LETTER_TO_CLUSTER[l] for l in y_letter])
    print(f"Loaded {len(records)} total usable samples across {len(set(y_letter))} classes.")

    # Fixed Stratified Train/Test Split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    records_arr = np.array(records, dtype=object)
    train_idx, test_idx = next(sss.split(records_arr, y_letter))

    train_records = records_arr[train_idx].tolist()
    train_y_letter = y_letter[train_idx]
    test_records = records_arr[test_idx].tolist()
    test_y_letter = y_letter[test_idx]

    print(f"Train split: {len(train_records)} samples | Test split: {len(test_records)} samples")

    # 1. Evaluate Configuration A: BASELINE
    baseline_res = evaluate_configuration(
        "Baseline",
        train_records,
        train_y_letter,
        test_records,
        test_y_letter,
        records,
        y_letter,
        y_cluster,
        augment=False,
        aug_copies=0,
        aug_params=aug_params,
        seed=args.seed,
        folds=args.folds,
    )

    # 2. Evaluate Configuration B: AUGMENTED
    augmented_res = evaluate_configuration(
        "Augmented",
        train_records,
        train_y_letter,
        test_records,
        test_y_letter,
        records,
        y_letter,
        y_cluster,
        augment=True,
        aug_copies=args.aug_copies,
        aug_params=aug_params,
        seed=args.seed,
        folds=args.folds,
    )

    # 3. Side-by-side Comparative Report
    print("\n" + "=" * 75)
    print("  OVERALL COMPARATIVE RESULTS")
    print("=" * 75)
    print(f"{'Metric':<25}{'Baseline':<16}{'Augmented':<16}{'Delta':<12}")
    print("-" * 75)
    d_test_acc = augmented_res["test_acc"] - baseline_res["test_acc"]
    d_test_f1 = augmented_res["test_f1"] - baseline_res["test_f1"]
    d_train_acc = augmented_res["train_acc"] - baseline_res["train_acc"]
    d_train_f1 = augmented_res["train_f1"] - baseline_res["train_f1"]
    d_cv = augmented_res["cv_mean"] - baseline_res["cv_mean"]

    print(f"{'Test Accuracy':<25}{baseline_res['test_acc']*100:>6.2f}%         {augmented_res['test_acc']*100:>6.2f}%         {d_test_acc*100:>+5.2f}%")
    print(f"{'Test Macro F1':<25}{baseline_res['test_f1']:>7.4f}          {augmented_res['test_f1']:>7.4f}          {d_test_f1:>+7.4f}")
    print(f"{'Train Accuracy':<25}{baseline_res['train_acc']*100:>6.2f}%         {augmented_res['train_acc']*100:>6.2f}%         {d_train_acc*100:>+5.2f}%")
    print(f"{'Train Macro F1':<25}{baseline_res['train_f1']:>7.4f}          {augmented_res['train_f1']:>7.4f}          {d_train_f1:>+7.4f}")
    print(f"{'5-Fold E2E CV Mean':<25}{baseline_res['cv_mean']*100:>6.2f}%         {augmented_res['cv_mean']*100:>6.2f}%         {d_cv*100:>+5.2f}%")
    print("-" * 75)

    # 4. Focused inspection of G, P, and Ş
    target_letters = ["G", "P", "Ş"]
    print("\n" + "=" * 75)
    print("  FOCUSED INSPECTION: G, P, and Ş")
    print("=" * 75)
    print(f"{'Letter':<8}{'Metric':<12}{'Baseline':<14}{'Augmented':<14}{'Delta':<10}")
    print("-" * 75)
    for letter in target_letters:
        bp = baseline_res["per_class"].get(letter, {})
        ap = augmented_res["per_class"].get(letter, {})
        prec_b, prec_a = bp.get("precision", 0.0), ap.get("precision", 0.0)
        rec_b, rec_a = bp.get("recall", 0.0), ap.get("recall", 0.0)
        f1_b, f1_a = bp.get("f1", 0.0), ap.get("f1", 0.0)
        supp = bp.get("support", 0)

        print(f"{letter:<8}{'Precision':<12}{prec_b:>6.2f}        {prec_a:>6.2f}        {prec_a - prec_b:>+5.2f}")
        print(f"{'':<8}{'Recall':<12}{rec_b:>6.2f}        {rec_a:>6.2f}        {rec_a - rec_b:>+5.2f}")
        print(f"{'':<8}{'F1-Score':<12}{f1_b:>6.2f}        {f1_a:>6.2f}        {f1_a - f1_b:>+5.2f} (N={supp})")
        print("-" * 50)

    # 5. Confusion analysis for G, P, Ş
    print("\n" + "=" * 75)
    print("  CONFUSION BREAKDOWN FOR G, P, Ş")
    print("=" * 75)
    classes = baseline_res["classes"]
    cm_base = np.array(baseline_res["confusion_matrix"])
    cm_aug = np.array(augmented_res["confusion_matrix"])

    for target in target_letters:
        if target not in classes:
            continue
        t_idx = classes.index(target)
        row_b = cm_base[t_idx]
        row_a = cm_aug[t_idx]

        print(f"\nGround Truth: {target} (Total test instances = {int(np.sum(row_b))})")
        print(f"  {'Predicted':<12}{'Baseline':<12}{'Augmented':<12}")
        confused_indices = sorted(
            list(set(np.where(row_b > 0)[0]).union(set(np.where(row_a > 0)[0])))
        )
        for c_idx in confused_indices:
            pred_letter = classes[c_idx]
            match_mark = " (CORRECT)" if pred_letter == target else ""
            print(f"  {pred_letter:<12}{int(row_b[c_idx]):>4}        {int(row_a[c_idx]):>4}{match_mark}")

    # 6. Save report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hyperparameters": {
            "seed": args.seed,
            "test_size": args.test_size,
            "max_per_class": args.max_per_class,
            "aug_copies": args.aug_copies,
            "aug_params": aug_params,
        },
        "summary": {
            "baseline_test_acc": baseline_res["test_acc"],
            "augmented_test_acc": augmented_res["test_acc"],
            "delta_test_acc": d_test_acc,
            "baseline_test_f1": baseline_res["test_f1"],
            "augmented_test_f1": augmented_res["test_f1"],
            "delta_test_f1": d_test_f1,
            "baseline_cv_mean": baseline_res["cv_mean"],
            "augmented_cv_mean": augmented_res["cv_mean"],
            "delta_cv_mean": d_cv,
        },
        "baseline": baseline_res,
        "augmented": augmented_res,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nFull comparative report saved to: {args.output}")


if __name__ == "__main__":
    main()


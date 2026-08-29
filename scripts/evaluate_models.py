#!/usr/bin/env python3
"""
Comprehensive Evaluation & Benchmarking Script for AzSL Fingerspelling System.

Evaluates:
1. Static letter classifier across the 32 AzSL alphabet letters
2. Dynamic sequence model across the 7 dynamic classes (D, Ü, Y, Ö, Z, C, Ş)
3. Integrated pipeline response times and decision arbitration
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.dynamic_dataset import (
    CLASS_TO_IDX,
    DYNAMIC_CLASSES,
    IDX_TO_CLASS,
    DynamicDatasetBuilder,
)
from scripts.dynamic_model import DynamicGestureRecognizer
from scripts.integrated_system import IntegratedSignRecognizer
from scripts.static_model import AZ_ALPHABET, StaticHierarchicalModel


def evaluate_static_system(static_model_path: str, data_dir: str):
    print("\n" + "=" * 60)
    print("1. EVALUATING STATIC CLASSIFIER")
    print("=" * 60)

    model = StaticHierarchicalModel(static_model_path)
    report = model.model_data.get("crossValidation", {})
    e2e = report.get("endToEnd", {})

    print(f"Static Model Loaded: {static_model_path}")
    print(f"Alphabet Size:       {len(model.alphabet)} letters")
    print(f"Cluster Count:       {len(model.clusters)} clusters")
    if e2e:
        print(f"5-Fold E2E Accuracy: {e2e.get('mean', 0.0) * 100:.2f}% (per-fold: {e2e.get('perFold', [])})")

    l2_reports = report.get("level2ByCluster", {})
    print("\nPer-Cluster Cross-Validation:")
    for cid, cinfo in model.clusters.items():
        letters = "+".join(cinfo["letters"])
        cv_info = l2_reports.get(cid)
        acc_str = f"{cv_info['mean']*100:.1f}%" if cv_info else "N/A"
        print(f"  Cluster {cid} [{letters}]: {acc_str}")


def evaluate_dynamic_system(checkpoint_path: str, cache_path: str, data_dir: str):
    print("\n" + "=" * 60)
    print("2. EVALUATING DYNAMIC SEQUENCE MODEL")
    print("=" * 60)

    if not os.path.isfile(checkpoint_path):
        print(f"[WARN] Dynamic model checkpoint not found at {checkpoint_path}. Train it first using scripts/train_dynamic.py")
        return

    recognizer = DynamicGestureRecognizer(checkpoint_path=checkpoint_path, device="cpu")
    print(f"Dynamic Model Loaded: {checkpoint_path}")
    print(f"Architecture:         {recognizer.config.get('model_type', 'gru').upper()} (hidden={recognizer.config.get('hidden_dim', 64)}, layers={recognizer.config.get('num_layers', 2)})")
    print(f"Sequence Length:      {recognizer.config.get('sequence_length', 20)} frames")
    print(f"Classes:              {recognizer.classes}")

    # Load evaluation report if available
    report_path = os.path.join(os.path.dirname(checkpoint_path), "dynamic_eval_report.json")
    if os.path.isfile(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            eval_report = json.load(f)
        metrics = eval_report.get("metrics", {})
        print(f"\nTest Accuracy:  {metrics.get('test_accuracy', 0.0) * 100:.2f}%")
        print(f"Test Macro F1:  {metrics.get('test_macro_f1', 0.0):.4f}")

        cm = eval_report.get("confusion_matrix")
        if cm:
            print("\nConfusion Matrix (7 Dynamic Classes):")
            header = "     " + " ".join([f"{c:>6}" for c in DYNAMIC_CLASSES])
            print(header)
            for i, row in enumerate(cm):
                row_str = f"{DYNAMIC_CLASSES[i]:<4} " + " ".join([f"{val:>6d}" for val in row])
                print(row_str)


def benchmark_latency(integrated_system: IntegratedSignRecognizer):
    print("\n" + "=" * 60)
    print("3. INFERENCE LATENCY BENCHMARK")
    print("=" * 60)

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Warmup
    for _ in range(5):
        integrated_system.process_frame(dummy_frame)

    n_runs = 50
    t0 = time.time()
    for _ in range(n_runs):
        integrated_system.process_frame(dummy_frame)
    elapsed = time.time() - t0
    avg_ms = (elapsed / n_runs) * 1000.0

    print(f"Integrated pipeline average processing time: {avg_ms:.2f} ms per frame (~{1000.0/max(0.1, avg_ms):.1f} FPS)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AzSL static and dynamic models")
    parser.add_argument("--static-model", default="public/models/azsl_hierarchical_model.json")
    parser.add_argument("--dynamic-model", default="models/dynamic_model.pt")
    parser.add_argument("--data-dir", default="data/AzSLD_Fingerspelling")
    parser.add_argument("--cache-path", default="data/dynamic_landmarks_cache.npz")
    args = parser.parse_args()

    evaluate_static_system(args.static_model, args.data_dir)
    evaluate_dynamic_system(args.dynamic_model, args.cache_path, args.data_dir)

    integrated = IntegratedSignRecognizer(
        static_model_path=args.static_model,
        dynamic_model_path=args.dynamic_model,
        mode="auto",
    )
    benchmark_latency(integrated)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Comprehensive Diagnostic Script for AzSL Dynamic Recognition Pipeline.
Analyzes:
1. Motion energy distributions across real dynamic sequences.
2. Dynamic model accuracy and confidence distributions.
3. Static classifier predictions and confidences on dynamic frames (static confusion/fallbacks).
4. Frame-by-frame motion gate simulation at various thresholds (0.038, 0.025, 0.015, etc.).
5. Buffer filling and temporal arbitration dynamics.
"""

import json
import os
import sys
from typing import Dict, List, Tuple

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from scripts.dynamic_dataset import DYNAMIC_CLASSES
from scripts.dynamic_model import DynamicGestureRecognizer
from scripts.static_model import StaticHierarchicalModel


def run_diagnostics():
    cache_path = "data/dynamic_landmarks_cache.npz"
    if not os.path.isfile(cache_path):
        print(f"Error: cache not found at {cache_path}")
        return

    cache = np.load(cache_path, allow_pickle=True)
    X = cache["X"]  # (893, 20, 63)
    y = cache["y"]  # (893,)

    static_model = StaticHierarchicalModel("public/models/azsl_hierarchical_model.json")
    dynamic_recognizer = DynamicGestureRecognizer("models/dynamic_model.pt", device="cpu")

    print("========================================================================================")
    print("1. MOTION ENERGY & DYNAMIC RECOGNITION DIAGNOSTIC (ACROSS ALL 893 REAL SEQUENCES)")
    print("========================================================================================")

    # Frame-to-frame mean absolute delta across 63 coords
    # energy = (1/63) * sum(|frame_t - frame_{t-1}|)
    # matching calculateMotionEnergy() in src/dynamicInference.js exactly.

    class_data = {
        c: {
            "mean_energies": [],
            "max_energies": [],
            "all_frame_energies": [],
            "dyn_confidences": [],
            "dyn_predictions": [],
            "static_predictions_all_frames": [],
            "static_confidences_all_frames": [],
            "static_predictions_final_frame": [],
            "static_confidences_final_frame": [],
        }
        for c in DYNAMIC_CLASSES
    }

    for i in range(len(X)):
        cls_name = DYNAMIC_CLASSES[y[i]]
        seq = X[i]  # (20, 63)

        frame_energies = []
        for t in range(1, 20):
            e = float(np.mean(np.abs(seq[t] - seq[t - 1])))
            frame_energies.append(e)
            class_data[cls_name]["all_frame_energies"].append(e)

        class_data[cls_name]["mean_energies"].append(np.mean(frame_energies))
        class_data[cls_name]["max_energies"].append(np.max(frame_energies))

        # Dynamic inference on the full 20-frame sequence
        dyn_res = dynamic_recognizer.predict_sequence(seq)
        class_data[cls_name]["dyn_predictions"].append(dyn_res["label"])
        class_data[cls_name]["dyn_confidences"].append(dyn_res["confidence"])

        # Static inference on individual frames of the sequence
        for t in range(20):
            raw_21x3 = seq[t].reshape(21, 3)
            # Static predict
            s_res = static_model.predict_from_landmarks(raw_21x3, mirror_x=False, velocity_xy=np.zeros(2))
            class_data[cls_name]["static_predictions_all_frames"].append(s_res["label"])
            class_data[cls_name]["static_confidences_all_frames"].append(s_res["confidence"])
            if t == 19:
                class_data[cls_name]["static_predictions_final_frame"].append(s_res["label"])
                class_data[cls_name]["static_confidences_final_frame"].append(s_res["confidence"])

    # Table 1: Motion Energy Ranges and Dynamic Model Accuracy
    print("%-6s | %-5s | %-19s | %-19s | %-16s | %-16s" % (
        "Class", "Count", "Mean Delta (min-max)", "Peak Delta (mean)", "Dyn Acc / Conf", "Static Fallback (on frames)"
    ))
    print("-" * 96)

    for c in DYNAMIC_CLASSES:
        d = class_data[c]
        count = len(d["mean_energies"])
        if count == 0:
            continue

        mean_e_avg = float(np.mean(d["mean_energies"]))
        mean_e_min = float(np.min(d["mean_energies"]))
        mean_e_max = float(np.max(d["mean_energies"]))
        peak_e_avg = float(np.mean(d["max_energies"]))

        dyn_acc = float(np.mean([1 if p == c else 0 for p in d["dyn_predictions"]])) * 100
        dyn_conf = float(np.mean(d["dyn_confidences"]))

        # Most common static prediction across dynamic frames
        s_preds = d["static_predictions_all_frames"]
        s_confs = d["static_confidences_all_frames"]
        unique, counts = np.unique(s_preds, return_counts=True)
        top_idx = np.argmax(counts)
        top_stat = unique[top_idx]
        top_stat_pct = counts[top_idx] / len(s_preds) * 100
        top_stat_conf = np.mean([sc for sp, sc in zip(s_preds, s_confs) if sp == top_stat])

        mean_str = f"{mean_e_avg:.4f} ({mean_e_min:.3f}-{mean_e_max:.3f})"
        peak_str = f"{peak_e_avg:.4f}"
        dyn_str = f"{dyn_acc:.1f}% ({dyn_conf:.2f})"
        stat_str = f"{top_stat} ({top_stat_pct:.0f}%, conf={top_stat_conf:.2f})"

        print("%-6s | %-5d | %-19s | %-19s | %-16s | %-16s" % (
            c, count, mean_str, peak_str, dyn_str, stat_str
        ))

    print("\n========================================================================================")
    print("2. MOTION GATE TRIGGER RATE AT DIFFERENT THRESHOLDS")
    print("========================================================================================")

    test_thresholds = [0.045, 0.038, 0.025, 0.020, 0.015, 0.010]
    header = "%-6s | " + " | ".join([f"Thresh {th:.3f}" for th in test_thresholds])
    print(header)
    print("-" * len(header))

    for c in DYNAMIC_CLASSES:
        d = class_data[c]
        row_strs = []
        for th in test_thresholds:
            # Fraction of sequences whose mean energy >= threshold
            seq_passed = np.mean([1 if e >= th else 0 for e in d["mean_energies"]]) * 100
            row_strs.append(f"{seq_passed:5.1f}%")
        print(f"%-6s | " % c + " | ".join(row_strs))

    print("\n========================================================================================")
    print("3. STATIC FALLBACK BREAKDOWN PER DYNAMIC CLASS")
    print("========================================================================================")
    for c in DYNAMIC_CLASSES:
        d = class_data[c]
        s_preds = d["static_predictions_all_frames"]
        unique, counts = np.unique(s_preds, return_counts=True)
        sort_indices = np.argsort(counts)[::-1]
        top_fallbacks = [
            f"{unique[idx]}: {counts[idx]/len(s_preds)*100:.1f}%"
            for idx in sort_indices[:4]
        ]
        print(f"Dynamic '{c}' static interpretations: {', '.join(top_fallbacks)}")

    print("\n========================================================================================")
    print("4. SIMULATING FRAME-BY-FRAME LIVE STREAMING ARBITRATION")
    print("========================================================================================")
    # Test streaming behavior with 20-frame buffer
    # When a user signs letter D or Z, they move hand for ~15-25 frames.
    # At start of gesture (t=0..10), buffer has old static frames + new motion frames.
    # Motion energy fluctuates: at direction changes (e.g. apex of arc for D, corner of Z),
    # instantaneous velocity drops to near 0!


if __name__ == "__main__":
    run_diagnostics()


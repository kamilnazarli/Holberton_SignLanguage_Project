#!/usr/bin/env python3
"""
Detailed diagnostic script for the 7 dynamic classes + static M.
"""

import sys, os
sys.path.insert(0, ".")
import numpy as np

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from scripts.dynamic_dataset import DYNAMIC_CLASSES
from scripts.dynamic_model import DynamicGestureRecognizer
from scripts.static_model import StaticHierarchicalModel

cache = np.load("data/dynamic_landmarks_cache.npz", allow_pickle=True)
X = cache["X"]
y = cache["y"]

recognizer = DynamicGestureRecognizer("models/dynamic_model.pt", device="cpu")
static_model = StaticHierarchicalModel("public/models/azsl_hierarchical_model.json")

print("==========================================================================================")
print("CLASS-BY-CLASS PROFILE: DYNAMIC SIGNATURES & MOTION CHARACTERISTICS")
print("==========================================================================================")

class_profiles = {}

for c_idx, c_name in enumerate(DYNAMIC_CLASSES):
    indices = np.where(y == c_idx)[0]
    count = len(indices)
    
    # 1. Motion energy over time (mean across sequences at each of the 19 frame deltas)
    # 2. Cumulative motion
    # 3. Trajectory spatial span
    # 4. Moving landmarks
    seqs = X[indices] # (N, 20, 63)
    
    # delta per step: (N, 19)
    step_deltas = np.mean(np.abs(np.diff(seqs, axis=1)), axis=2) # (N, 19)
    mean_step_delta = np.mean(step_deltas, axis=0) # (19,)
    
    # cumulative displacement:
    cum_displacements = np.sum(np.linalg.norm(np.diff(seqs, axis=1), axis=2), axis=1) # (N,)
    
    # net window span: ||frame_19 - frame_0||
    net_spans = np.linalg.norm(seqs[:, 19, :] - seqs[:, 0, :], axis=1) # (N,)
    
    # Dynamic model confidence
    confs = []
    preds = []
    for s in seqs:
        res = recognizer.predict_sequence(s)
        preds.append(res["label"])
        confs.append(res["confidence"])
        
    acc = np.mean([1 if p == c_name else 0 for p in preds]) * 100
    
    # Static model predictions on frame 19 (final frame) and frame 10 (mid frame)
    static_preds = []
    for s in seqs:
        raw_mid = s[10].reshape(21, 3)
        s_res = static_model.predict_from_landmarks(raw_mid, mirror_x=False, velocity_xy=np.zeros(2))
        static_preds.append(s_res["label"])
    top_static = dict(zip(*np.unique(static_preds, return_counts=True)))
    
    class_profiles[c_name] = {
        "count": count,
        "mean_step_delta": mean_step_delta,
        "avg_cum_disp": float(np.mean(cum_displacements)),
        "min_cum_disp": float(np.min(cum_displacements)),
        "avg_net_span": float(np.mean(net_spans)),
        "dyn_acc": acc,
        "avg_conf": float(np.mean(confs)),
        "top_static": top_static,
    }
    
    print(f"--- Class: {c_name} (N={count} sequences) ---")
    print(f"  Dynamic GRU Accuracy  : {acc:.1f}% (Avg Confidence: {np.mean(confs):.3f})")
    print(f"  Cumulative Displacement: {np.mean(cum_displacements):.2f} (min={np.min(cum_displacements):.2f}, max={np.max(cum_displacements):.2f})")
    print(f"  Net Spatial Span (end-to-end): {np.mean(net_spans):.2f}")
    print(f"  Static Interpretation (mid-gesture frame 10): {top_static}")
    print(f"  Energy profile across 19 steps (min={np.min(mean_step_delta):.3f}, max={np.max(mean_step_delta):.3f}, mean={np.mean(mean_step_delta):.3f})")
    print()


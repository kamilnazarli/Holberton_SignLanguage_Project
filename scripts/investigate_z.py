#!/usr/bin/env python3
"""
Deep investigation into Z class detection and confusion.
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

z_indices = np.where(y == DYNAMIC_CLASSES.index("Z"))[0]
print(f"=== ANALYZING ALL {len(z_indices)} Z SEQUENCES ===")

# Test 1: What does the dynamic GRU output on all 102 Z sequences?
gru_preds = []
gru_confs = []
for idx in z_indices:
    res = recognizer.predict_sequence(X[idx])
    gru_preds.append(res["label"])
    gru_confs.append(res["confidence"])

print(f"Full-window GRU Accuracy on Z: {np.mean([1 if p == 'Z' else 0 for p in gru_preds])*100:.1f}%")
print(f"Average Confidence on Z: {np.mean(gru_confs):.4f}")

# Test 2: What does static model predict on the frames of Z?
# Landmark 8 is index tip (x=24, y=25, z=26)
# Let's inspect the actual shape of Z gestures in the dataset
# Where does Z move? Does the wrist move or does the finger move?
for i in range(min(5, len(z_indices))):
    seq = X[z_indices[i]]
    # Check index fingertip (landmark 8)
    tip_x = seq[:, 24]
    tip_y = seq[:, 25]
    print(f"Z sample {i}:")
    print(f"  Tip X: min={np.min(tip_x):.2f}, max={np.max(tip_x):.2f}, start={tip_x[0]:.2f}, mid={tip_x[10]:.2f}, end={tip_x[19]:.2f}")
    print(f"  Tip Y: min={np.min(tip_y):.2f}, max={np.max(tip_y):.2f}, start={tip_y[0]:.2f}, mid={tip_y[10]:.2f}, end={tip_y[19]:.2f}")

# Test 3: What does the GRU predict if Z is windowed with trailing stationary frames?
# i.e. When user completes the Z gesture, the buffer contains Z gesture in the first 10 frames and stationary hand in the last 10 frames!
print("\n=== GRU PREDICTION ON POST-GESTURE WINDOW (10 GESTURE FRAMES + 10 SETTLING FRAMES) ===")
for i in range(min(5, len(z_indices))):
    seq = X[z_indices[i]]
    # 10 gesture frames followed by 10 stationary frames of the final pose
    tail_window = np.concatenate([seq[10:20], np.repeat(seq[-1:], 10, axis=0)], axis=0)
    head_window = np.concatenate([np.repeat(seq[0:1], 10, axis=0), seq[0:10]], axis=0)
    r_tail = recognizer.predict_sequence(tail_window)
    r_head = recognizer.predict_sequence(head_window)
    print(f"Z sample {i}: Head window -> {r_head['label']} ({r_head['confidence']:.3f}) | Tail window -> {r_tail['label']} ({r_tail['confidence']:.3f})")

# Test 4: What about D under head/tail windows?
print("\n=== GRU PREDICTION ON D POST-GESTURE WINDOW ===")
d_indices = np.where(y == DYNAMIC_CLASSES.index("D"))[0]
for i in range(min(5, len(d_indices))):
    seq = X[d_indices[i]]
    tail_window = np.concatenate([seq[10:20], np.repeat(seq[-1:], 10, axis=0)], axis=0)
    head_window = np.concatenate([np.repeat(seq[0:1], 10, axis=0), seq[0:10]], axis=0)
    r_tail = recognizer.predict_sequence(tail_window)
    r_head = recognizer.predict_sequence(head_window)
    print(f"D sample {i}: Head window -> {r_head['label']} ({r_head['confidence']:.3f}) | Tail window -> {r_tail['label']} ({r_tail['confidence']:.3f})")


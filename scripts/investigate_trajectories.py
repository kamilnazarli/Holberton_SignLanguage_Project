#!/usr/bin/env python3
"""
Deep investigation into Z, D, Y, M and all 7 dynamic classes.
"""

import sys, os, glob
sys.path.insert(0, ".")
import numpy as np

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from scripts.dynamic_dataset import DYNAMIC_CLASSES
from scripts.dynamic_model import DynamicGestureRecognizer
from scripts.static_model import StaticHierarchicalModel

cache = np.load("data/dynamic_landmarks_cache.npz", allow_pickle=True)
X = cache["X"] # (893, 20, 63)
y = cache["y"] # (893,)

recognizer = DynamicGestureRecognizer("models/dynamic_model.pt", device="cpu")
static_model = StaticHierarchicalModel("public/models/azsl_hierarchical_model.json")

print("========================================================================================")
print("INVESTIGATION A: Z GESTURE TRAJECTORY AND MOTION PATTERNS")
print("========================================================================================")
z_indices = np.where(y == DYNAMIC_CLASSES.index("Z"))[0]
print(f"Total Z sequences in cache: {len(z_indices)}")

# Check index finger tip (landmark 8: indices 24, 25, 26 in 63D) trajectory for Z
# In 63D: landmark 8 is x=24, y=25, z=26
z_seqs = X[z_indices]
for idx in range(min(5, len(z_indices))):
    seq = z_seqs[idx]
    tip_x = seq[:, 24]
    tip_y = seq[:, 25]
    total_path = np.sum(np.sqrt(np.diff(tip_x)**2 + np.diff(tip_y)**2))
    wrist_x = seq[:, 0]
    wrist_y = seq[:, 1]
    wrist_path = np.sum(np.sqrt(np.diff(wrist_x)**2 + np.diff(wrist_y)**2))
    print(f"Z sample {idx}: Index Tip Path Length = {total_path:.3f}, Wrist Path = {wrist_path:.3f}")

print("\n========================================================================================")
print("INVESTIGATION B: D GESTURE TRAJECTORY AND MOTION PATTERNS")
print("========================================================================================")
d_indices = np.where(y == DYNAMIC_CLASSES.index("D"))[0]
d_seqs = X[d_indices]
for idx in range(min(5, len(d_indices))):
    seq = d_seqs[idx]
    tip_x = seq[:, 24]
    tip_y = seq[:, 25]
    total_path = np.sum(np.sqrt(np.diff(tip_x)**2 + np.diff(tip_y)**2))
    wrist_x = seq[:, 0]
    wrist_y = seq[:, 1]
    wrist_path = np.sum(np.sqrt(np.diff(wrist_x)**2 + np.diff(wrist_y)**2))
    print(f"D sample {idx}: Index Tip Path Length = {total_path:.3f}, Wrist Path = {wrist_path:.3f}")

print("\n========================================================================================")
print("INVESTIGATION C: PATH LENGTH / CUMULATIVE DISPLACEMENT VS INSTANTANEOUS DELTA")
print("========================================================================================")
# Check cumulative displacement across the 20 frames for each class
for c in DYNAMIC_CLASSES:
    c_idx = DYNAMIC_CLASSES.index(c)
    indices = np.where(y == c_idx)[0]
    paths = []
    max_spans = [] # max distance between any two frames in the 20-frame window
    for i in indices:
        seq = X[i]
        # Cumulative displacement across 63 features
        frame_diffs = np.linalg.norm(np.diff(seq, axis=0), axis=1) # (19,)
        total_disp = np.sum(frame_diffs)
        # Max span across the window: norm(seq[19] - seq[0]) or max pairwise
        span = np.linalg.norm(seq[-1] - seq[0])
        paths.append(total_disp)
        max_spans.append(span)
    print(f"Class {c:2s}: Cumulative Displacement = {np.mean(paths):.3f} (min={np.min(paths):.3f}, max={np.max(paths):.3f}) | Net Window Span = {np.mean(max_spans):.3f}")

print("\n========================================================================================")
print("INVESTIGATION D: WHAT HAPPENS TO STATIC LETTERS (CUMULATIVE DISPLACEMENT)")
print("========================================================================================")
# If a static letter is held still with normal camera noise / tremor, what is cumulative displacement?
# Let's test with synthetic jitter (sigma=0.005, 0.010, 0.015)
for sigma in [0.003, 0.006, 0.010, 0.015]:
    dummy_static = np.zeros((20, 63))
    jitter = np.random.normal(0, sigma, dummy_static.shape)
    diffs = np.linalg.norm(np.diff(jitter, axis=0), axis=1)
    cum_disp = np.sum(diffs)
    span = np.linalg.norm(jitter[-1] - jitter[0])
    mean_abs = np.mean(np.abs(np.diff(jitter, axis=0)))
    print(f"Static with jitter sigma={sigma:.3f}: Mean Abs Delta = {mean_abs:.4f}, Cumulative Disp = {cum_disp:.3f}, Net Span = {span:.3f}")


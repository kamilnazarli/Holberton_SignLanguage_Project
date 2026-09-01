#!/usr/bin/env python3
"""
Deep investigation into why Z is confused with Y.
Tests all 102 Z sequences under:
- Truncated windows (first 10 frames, middle 10 frames, last 10 frames)
- Downsampled / stretched windows
- Mirrored landmarks (Left hand vs Right hand)
- Different speed / sampling rates
- What is the difference between Z and Y in feature space?
"""

import sys, os
sys.path.insert(0, ".")
import numpy as np

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from scripts.dynamic_dataset import DYNAMIC_CLASSES, resample_sequence
from scripts.dynamic_model import DynamicGestureRecognizer

cache = np.load("data/dynamic_landmarks_cache.npz", allow_pickle=True)
X = cache["X"]
y = cache["y"]

recognizer = DynamicGestureRecognizer("models/dynamic_model.pt", device="cpu")

z_indices = np.where(y == DYNAMIC_CLASSES.index("Z"))[0]
y_indices = np.where(y == DYNAMIC_CLASSES.index("Y"))[0]

print("==========================================================================================")
print("1. EVALUATING Z UNDER TEMPORAL TRUNCATIONS & PARTIAL SLIDING WINDOWS")
print("==========================================================================================")
# In real life, a user draws Z in ~10-15 frames.
# What does the model output when the 20-frame buffer contains:
# - Z gesture in frames 0..10, followed by resting in frames 11..19?
# - Resting in frames 0..9, followed by Z gesture in frames 10..19?
# - Faster Z gesture (resampled from 8, 10, 12, 14 frames)?

confused_as_y = 0
confused_as_other = {}
total_tests = 0

for idx in z_indices:
    seq = X[idx] # (20, 63)
    
    # Test sub-window 1: First 12 frames padded with final pose
    w_early = np.concatenate([seq[:12], np.repeat(seq[11:12], 8, axis=0)], axis=0)
    # Test sub-window 2: Last 12 frames preceded by initial pose
    w_late = np.concatenate([np.repeat(seq[0:1], 8, axis=0), seq[8:]], axis=0)
    # Test sub-window 3: Middle 12 frames
    w_mid = np.concatenate([np.repeat(seq[4:5], 4, axis=0), seq[4:16], np.repeat(seq[15:16], 4, axis=0)], axis=0)
    # Test sub-window 4: Resampled from 10 frames
    w_fast = resample_sequence(seq[::2], 20)
    
    for w_name, w in [("early", w_early), ("late", w_late), ("mid", w_mid), ("fast", w_fast)]:
        res = recognizer.predict_sequence(w)
        total_tests += 1
        lbl = res["label"]
        if lbl == "Y":
            confused_as_y += 1
        elif lbl != "Z":
            confused_as_other[lbl] = confused_as_other.get(lbl, 0) + 1

print(f"Total window tests: {total_tests}")
print(f"Confused as Y: {confused_as_y} ({confused_as_y / total_tests * 100:.1f}%)")
print(f"Confused as others: {confused_as_other}")

# Inspect a specific sequence that got predicted as Y
print("\n--- Examining sequences that predict Y ---")
for idx in z_indices[:20]:
    seq = X[idx]
    w_late = np.concatenate([np.repeat(seq[0:1], 10, axis=0), seq[10:]], axis=0)
    res = recognizer.predict_sequence(w_late)
    if res["label"] == "Y":
        probs = res["probabilities"]
        print(f"Z sample {idx} (late window) -> Predicted: {res['label']} ({res['confidence']:.3f}) | P(Z)={probs.get('Z',0):.3f}, P(Y)={probs.get('Y',0):.3f}")

print("\n==========================================================================================")
print("2. WHAT DOES Y ACTUALLY LOOK LIKE VS Z?")
print("==========================================================================================")
# For Y: In AzSL, Y is formed with thumb and pinky extended, moving downward.
# For Z: In AzSL, Z is index finger drawing a Z.
# Let's inspect the hand shapes of Y vs Z in the dataset:
y_sample = X[y_indices[0]] # (20, 63)
z_sample = X[z_indices[0]] # (20, 63)

# Check finger extension: Index tip (lm 8) vs Pinky tip (lm 20) vs Thumb tip (lm 4)
# In 63D:
# Wrist = 0..2
# Thumb tip (lm 4) = 12..14
# Index tip (lm 8) = 24..26
# Pinky tip (lm 20) = 60..62
print("Z Sample 0:")
print(f"   Thumb tip Y : {z_sample[0, 13]:.2f} -> {z_sample[19, 13]:.2f}")
print(f"   Index tip Y : {z_sample[0, 25]:.2f} -> {z_sample[19, 25]:.2f}")
print(f"   Pinky tip Y : {z_sample[0, 61]:.2f} -> {z_sample[19, 61]:.2f}")

print("\nY Sample 0:")
print(f"   Thumb tip Y : {y_sample[0, 13]:.2f} -> {y_sample[19, 13]:.2f}")
print(f"   Index tip Y : {y_sample[0, 25]:.2f} -> {y_sample[19, 25]:.2f}")
print(f"   Pinky tip Y : {y_sample[0, 61]:.2f} -> {y_sample[19, 61]:.2f}")

# Now check what happens if someone signs Z with the wrong hand or if mirror_x is flipped!
print("\n==========================================================================================")
print("3. WHAT HAPPENS IF Z IS FLIPPED IN X (MIRRORED)?")
print("==========================================================================================")
for idx in z_indices[:5]:
    seq = X[idx].copy()
    # Flip X coordinates (every 3rd index starting at 0: 0, 3, 6, ..., 60)
    seq_flipped = seq.copy()
    seq_flipped[:, 0::3] *= -1.0
    res_normal = recognizer.predict_sequence(seq)
    res_flipped = recognizer.predict_sequence(seq_flipped)
    print(f"Z sample {idx}: Normal -> {res_normal['label']} ({res_normal['confidence']:.2f}) | Flipped -> {res_flipped['label']} ({res_flipped['confidence']:.2f})")


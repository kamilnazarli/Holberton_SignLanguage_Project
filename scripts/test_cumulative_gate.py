#!/usr/bin/env python3
"""
Test dual arbitration with cumulative motion gating.
"""

import sys, os, glob
sys.path.insert(0, ".")
import numpy as np
import cv2

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from scripts.dynamic_dataset import DYNAMIC_CLASSES, LandmarkerWrapper, normalize_landmarks_63
from scripts.dynamic_model import DynamicGestureRecognizer
from scripts.static_model import StaticHierarchicalModel

cache = np.load("data/dynamic_landmarks_cache.npz", allow_pickle=True)
X = cache["X"]
y = cache["y"]

recognizer = DynamicGestureRecognizer("models/dynamic_model.pt", device="cpu")
static_model = StaticHierarchicalModel("public/models/azsl_hierarchical_model.json")
landmarker = LandmarkerWrapper("public/models/hand_landmarker.task", min_confidence=0.5)

CUMULATIVE_THRESH = 4.8

print("=== TESTING DUAL PIPELINE WITH CUMULATIVE MOTION GATE (THRESH=4.8) ===")

print("--- STATIC LETTERS (M, B, S, K, T, A) ---")
static_test_letters = ["M", "B", "S", "K", "T", "A"]
for let in static_test_letters:
    files = sorted(glob.glob(f"data/AzSLD_Fingerspelling/{let}/*.jpg"))[:5]
    for f in files:
        img = cv2.imread(f)
        if img is None: continue
        res = landmarker.extract_landmarks(img)
        if res is None: continue
        xyz, mirror_x = res
        s_res = static_model.predict_from_landmarks(xyz, mirror_x=mirror_x)
        norm63 = normalize_landmarks_63(xyz, mirror_x=mirror_x)
        # Add normal live tremor sigma=0.010
        jitter_seq = np.repeat(norm63.reshape(1, 63), 20, axis=0) + np.random.normal(0, 0.010, (20, 63)).astype(np.float32)
        cum_disp = float(np.sum(np.linalg.norm(np.diff(jitter_seq, axis=0), axis=1)))
        
        # Dual arbitration:
        if cum_disp >= CUMULATIVE_THRESH:
            d_res = recognizer.predict_sequence(jitter_seq)
            mode = "DYNAMIC"
            label = d_res["label"]
        else:
            mode = "STATIC"
            label = s_res["label"]
        status = "OK" if label == let else "FAIL"
        print(f"Static {let:2s}: CumDisp={cum_disp:.2f} -> Mode={mode:7s} Label={label:2s} [{status}]")

print("\n--- DYNAMIC LETTERS (ALL 7 CLASSES) ---")
for c in DYNAMIC_CLASSES:
    idx = DYNAMIC_CLASSES.index(c)
    indices = np.where(y == idx)[0][:15]
    successes = 0
    for i in indices:
        seq = X[i]
        cum_disp = float(np.sum(np.linalg.norm(np.diff(seq, axis=0), axis=1)))
        if cum_disp >= CUMULATIVE_THRESH:
            d_res = recognizer.predict_sequence(seq)
            mode = "DYNAMIC"
            label = d_res["label"]
        else:
            s_res = static_model.predict_from_landmarks(seq[10].reshape(21, 3), mirror_x=False, velocity_xy=np.zeros(2))
            mode = "STATIC"
            label = s_res["label"]
        if label == c: successes += 1
    print(f"Dynamic {c:2s}: {successes}/{len(indices)} correctly recognized as {c}")


#!/usr/bin/env python3
"""
Test rigid translation vs wrist-relative motion in Z, D, and dynamic letters.
"""

import sys, os
sys.path.insert(0, ".")
import numpy as np

from scripts.dynamic_dataset import DYNAMIC_CLASSES
from scripts.dynamic_model import DynamicGestureRecognizer

recognizer = DynamicGestureRecognizer("models/dynamic_model.pt", device="cpu")

# Take a stationary index-pointing-up pose (like frame 0 of D or Z)
cache = np.load("data/dynamic_landmarks_cache.npz", allow_pickle=True)
X = cache["X"]
y = cache["y"]

z_idx = np.where(y == DYNAMIC_CLASSES.index("Z"))[0][0]
d_idx = np.where(y == DYNAMIC_CLASSES.index("D"))[0][0]

z_sample = X[z_idx] # (20, 63)
d_sample = X[d_idx] # (20, 63)

# If a user holds index pointing up and moves entire hand in a Z pattern across the camera,
# but the normalized 63D landmarks subtract the wrist on every frame:
# Relative to wrist, the landmarks are completely rigid/stationary!
rigid_pointing = np.repeat(z_sample[0:1], 20, axis=0) # (20, 63)
res_rigid = recognizer.predict_sequence(rigid_pointing)
print(f"Rigid pointing finger (whole hand moving in camera, but wrist-normalized):")
print(f"  --> Predicted: {res_rigid['label']} (confidence: {res_rigid['confidence']:.3f})")
print(f"  --> Top candidates: {res_rigid['candidates']}")

# Now check what the genuine Z sample in the dataset does:
res_genuine_z = recognizer.predict_sequence(z_sample)
print(f"\nGenuine dataset Z sample:")
print(f"  --> Predicted: {res_genuine_z['label']} (confidence: {res_genuine_z['confidence']:.3f})")

# Let's inspect the motion of ALL 21 landmarks in z_sample:
landmark_stds = np.std(z_sample, axis=0) # (63,)
top_moving_landmarks = np.argsort(landmark_stds)[::-1][:6]
print(f"\nTop moving coordinates in dataset Z sample:")
for idx in top_moving_landmarks:
    lm_idx = idx // 3
    axis = ['X', 'Y', 'Z'][idx % 3]
    print(f"  Landmark {lm_idx:2d} {axis}: std={landmark_stds[idx]:.4f}")


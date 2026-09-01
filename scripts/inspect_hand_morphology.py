#!/usr/bin/env python3
"""
Inspect the physical hand morphology of H, P, C, Ç, K in the dataset.
"""

import sys, os, glob
sys.path.insert(0, ".")
import numpy as np
import cv2

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from scripts.dynamic_dataset import LandmarkerWrapper
from scripts.static_model import (
    LM,
    FINGERS,
    StaticHierarchicalModel,
    normalize_landmarks,
    joint_angles_15,
    tip_distances_4,
    build_feature_vector_84,
    apply_scaler,
    mlp_forward,
)

def imread_unicode(path):
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

landmarker = LandmarkerWrapper("public/models/hand_landmarker.task", min_confidence=0.5)
static_model = StaticHierarchicalModel("public/models/azsl_hierarchical_model.json")

letters = ["H", "P", "C", "Ç", "K"]

print("==========================================================================================")
print("HAND MORPHOLOGY IN THE DATASET (FINGER ANGLES & TIP DISTANCES)")
print("==========================================================================================")

for let in letters:
    folder = os.path.join("data", "AzSLD_Fingerspelling", let)
    files = sorted(glob.glob(os.path.join(folder, "*.*")))[:10]
    
    all_angles = []
    all_dists = []
    
    for f in files:
        img = imread_unicode(f)
        if img is None: continue
        res = landmarker.extract_landmarks(img)
        if res is None: continue
        xyz, mirror_x = res
        coords = normalize_landmarks(xyz, mirror_x=mirror_x)
        angles = joint_angles_15(coords)
        dists = tip_distances_4(coords)
        all_angles.append(angles)
        all_dists.append(dists)
        
    avg_angles = np.mean(all_angles, axis=0) # 15 angles: [thumb base, tip, spread, index b, t, s, middle b, t, s, ring b, t, s, pinky b, t, s]
    avg_dists = np.mean(all_dists, axis=0)   # 4 dists: [T-I, I-M, M-R, R-P]
    
    # Let's print index, middle, ring, pinky extension (tip flex angle: 180 = straight, small = curled)
    # in joint_angles_15:
    # index: base=angles[3], tip=angles[4], spread=angles[5]
    # middle: base=angles[6], tip=angles[7], spread=angles[8]
    # ring: base=angles[9], tip=angles[10], spread=angles[11]
    # pinky: base=angles[12], tip=angles[13], spread=angles[14]
    print(f"Letter '{let}':")
    print(f"   Index tip-flex : {avg_angles[4]:5.1f}° | Middle tip-flex: {avg_angles[7]:5.1f}° | Ring tip-flex: {avg_angles[10]:5.1f}° | Pinky tip-flex: {avg_angles[13]:5.1f}°")
    print(f"   Tip gaps (T-I, I-M, M-R, R-P): {[round(float(x), 2) for x in avg_dists]}")
    print()


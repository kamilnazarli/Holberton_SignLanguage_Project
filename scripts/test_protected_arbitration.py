#!/usr/bin/env python3
"""
Test Protected Arbitration with Cumulative Motion Gate across all letters.
"""

import sys, os, glob
sys.path.insert(0, ".")
import numpy as np

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from scripts.dynamic_dataset import DYNAMIC_CLASSES, LandmarkerWrapper, normalize_landmarks_63
from scripts.dynamic_model import DynamicGestureRecognizer
from scripts.static_model import StaticHierarchicalModel

STATIC_ONLY_CLASSES = set([
    'A', 'B', 'E', 'Ə', 'F', 'G', 'Ğ', 'H', 'X', 'I', 'İ',
    'J', 'K', 'Q', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V'
])

cache = np.load("data/dynamic_landmarks_cache.npz", allow_pickle=True)
X = cache["X"]
y = cache["y"]

recognizer = DynamicGestureRecognizer("models/dynamic_model.pt", device="cpu")
static_model = StaticHierarchicalModel("public/models/azsl_hierarchical_model.json")
landmarker = LandmarkerWrapper("public/models/hand_landmarker.task", min_confidence=0.5)

print("=== PART 1: TESTING STATIC LETTERS (M, B, S, K, T, A) WITH NEW PROTECTED ARBITRATION ===")
# Even if a static letter moves slightly (simulated with 0.025 frame delta tremor):
for let in ['M', 'B', 'S', 'K', 'T', 'A']:
    files = sorted(glob.glob(f"data/AzSLD_Fingerspelling/{let}/*.jpg"))[:5]
    for f in files:
        img = cv2.imread(f) if 'cv2' in globals() else None
        # load via PIL/cv2
        import cv2
        img = cv2.imread(f)
        if img is None: continue
        res = landmarker.extract_landmarks(img)
        if res is None: continue
        xyz, mirror_x = res
        
        # Static model
        s_res = static_model.predict_from_landmarks(xyz, mirror_x=mirror_x)
        
        # Jitter sequence (simulating hand moving while signing static letter)
        norm63 = normalize_landmarks_63(xyz, mirror_x=mirror_x)
        jitter_seq = np.repeat(norm63.reshape(1, 63), 20, axis=0) + np.random.normal(0, 0.012, (20, 63)).astype(np.float32)
        
        # Cumulative motion
        cum_disp = float(np.sum(np.linalg.norm(np.diff(jitter_seq, axis=0), axis=1)))
        
        # Dynamic model prediction
        d_res = recognizer.predict_sequence(jitter_seq)
        
        # Protected arbitration logic:
        # 1. Check if static model is confident on a static-only letter
        is_static_protected = (s_res['confidence'] >= 0.70 and s_res['label'] in STATIC_ONLY_CLASSES)
        
        # 2. Check if cumulative motion qualifies as a real dynamic gesture (> 7.0)
        has_dynamic_trajectory = (cum_disp >= 7.0)
        
        if has_dynamic_trajectory and not is_static_protected and d_res['confidence'] >= 0.65:
            decision_mode = "DYNAMIC"
            decision_label = d_res['label']
        else:
            decision_mode = "STATIC"
            decision_label = s_res['label']
            
        print(f"Letter {let:2s}: Static='{s_res['label']}' ({s_res['confidence']:.2f}) | Dyn='{d_res['label']}' ({d_res['confidence']:.2f}) | CumDisp={cum_disp:.2f} -> Decision: {decision_mode} '{decision_label}'")
    print()

print("=== PART 2: TESTING ALL 7 DYNAMIC CLASSES (REAL SEQUENCES) WITH PROTECTED ARBITRATION ===")
for c in DYNAMIC_CLASSES:
    idx = DYNAMIC_CLASSES.index(c)
    indices = np.where(y == idx)[0][:5]
    for i in indices:
        seq = X[i]
        cum_disp = float(np.sum(np.linalg.norm(np.diff(seq, axis=0), axis=1)))
        d_res = recognizer.predict_sequence(seq)
        
        # Mid frame static prediction
        s_res = static_model.predict_from_landmarks(seq[10].reshape(21, 3), mirror_x=False, velocity_xy=np.zeros(2))
        
        is_static_protected = (s_res['confidence'] >= 0.70 and s_res['label'] in STATIC_ONLY_CLASSES)
        has_dynamic_trajectory = (cum_disp >= 7.0)
        
        if has_dynamic_trajectory and not is_static_protected and d_res['confidence'] >= 0.65:
            decision_mode = "DYNAMIC"
            decision_label = d_res['label']
        else:
            decision_mode = "STATIC"
            decision_label = s_res['label']
            
        status = "SUCCESS" if decision_label == c else "FAIL"
        print(f"Dynamic {c:2s}: CumDisp={cum_disp:5.2f} | Dyn='{d_res['label']}' ({d_res['confidence']:.2f}) | Stat='{s_res['label']}' ({s_res['confidence']:.2f}) -> {decision_mode} '{decision_label}' [{status}]")
    print()


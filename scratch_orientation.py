import sys, os, glob
import numpy as np
import cv2

sys.stdout.reconfigure(encoding="utf-8")

from scripts.dynamic_dataset import LandmarkerWrapper
from scripts.static_model import (
    normalize_landmarks,
    build_feature_vector_84,
    apply_scaler,
    mlp_forward,
)

def imread_unicode(path):
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

landmarker = LandmarkerWrapper("public/models/hand_landmarker.task", min_confidence=0.5)

letters = ["G", "Ş", "H", "P", "J", "Ç", "F"]

print("==========================================================================================")
print("HAND ORIENTATION IN DATASET: Vector from WRIST (lm 0) to MIDDLE_MCP (lm 9)")
print("(In normalized coords: dx, dy. Note: dy < 0 means pointing UP, dy > 0 means pointing DOWN)")
print("==========================================================================================")

for letter in letters:
    folder = os.path.join("data", "AzSLD_Fingerspelling", letter)
    files = sorted(glob.glob(os.path.join(folder, "*.*")))[:15]
    
    palm_dirs = []
    index_dirs = []
    
    for f in files:
        img = imread_unicode(f)
        if img is None: continue
        res = landmarker.extract_landmarks(img)
        if res is None: continue
        xyz, mirror_x = res
        coords = normalize_landmarks(xyz, mirror_x=mirror_x)
        
        # In normalized coords:
        # Wrist is at (0, 0, 0)
        # Middle MCP is coords[9]
        # Index TIP is coords[8]
        palm_dir = coords[9] # vector from wrist to middle MCP
        index_dir = coords[8] # vector from wrist to index tip
        palm_dirs.append(palm_dir)
        index_dirs.append(index_dir)
        
    avg_palm = np.mean(palm_dirs, axis=0)
    avg_index = np.mean(index_dirs, axis=0)
    print(f"Letter {letter:2s} (avg of {len(palm_dirs)} samples):")
    print(f"   Palm vector (Wrist -> Mid_MCP): [x={avg_palm[0]:+5.2f}, y={avg_palm[1]:+5.2f}, z={avg_palm[2]:+5.2f}]")
    print(f"   Index tip   (Wrist -> Idx_TIP): [x={avg_index[0]:+5.2f}, y={avg_index[1]:+5.2f}, z={avg_index[2]:+5.2f}]")


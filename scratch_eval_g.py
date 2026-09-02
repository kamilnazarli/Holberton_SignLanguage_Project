import sys, os, glob
import json
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
with open("public/models/azsl_hierarchical_model.json", "r", encoding="utf-8") as f:
    model_json = json.load(f)

folder = os.path.join("data", "AzSLD_Fingerspelling", "G")
all_files = sorted(glob.glob(os.path.join(folder, "*.*")))
img_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

print(f"Total G images: {len(img_files)}")
step = max(1, len(img_files) // 10)
sample_files = img_files[::step][:10]

key_letters = ["G", "H", "J", "Ş", "P", "F", "Ç"]

for f in sample_files:
    fname = os.path.basename(f)
    img = imread_unicode(f)
    if img is None: continue
    res = landmarker.extract_landmarks(img)
    if res is None:
        print(f"File {fname}: No hand detected")
        continue
    raw_xyz, mirror_x = res
    coords = normalize_landmarks(raw_xyz, mirror_x=mirror_x)
    full84 = build_feature_vector_84(coords, velocity_xy=np.zeros(2, dtype=np.float64))
    
    l1_in = apply_scaler(full84, model_json["level1"]["scaler"])
    l1_cands = mlp_forward(model_json["level1"]["model"], l1_in)
    top_cluster = str(l1_cands[0][0])
    
    c_entry = model_json["clusters"][top_cluster]
    f_indices = c_entry["featureIndices"]
    sub_in = full84 if len(f_indices) == len(full84) else full84[f_indices]
    l2_in = apply_scaler(sub_in, c_entry["scaler"])
    l2_cands = mlp_forward(c_entry["model"], l2_in)
    
    dispatched_dist = {let: 0.0 for let in key_letters}
    for let, prob in l2_cands:
        if let in dispatched_dist:
            dispatched_dist[let] = prob
            
    print(f"Sample {fname}:")
    print(f"  Level 1: {[(str(c), round(p, 3)) for c, p in l1_cands[:2]]} -> Cluster {top_cluster}")
    print(f"  Top 5 in Cluster {top_cluster}: {[(let, round(p, 3)) for let, p in l2_cands[:5]]}")
    key_str = " | ".join([f"P({k})={dispatched_dist[k]:.3f}" for k in key_letters])
    print(f"  Key Probs: {key_str}\n")


import sys, os, glob
import json
import numpy as np
import cv2

sys.stdout.reconfigure(encoding="utf-8")

from scripts.dynamic_dataset import LandmarkerWrapper
from scripts.static_model import (
    StaticHierarchicalModel,
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

targets = ["G", "H", "J"]
key_letters = ["G", "H", "J", "Ş", "P", "F", "Ç"]

print("==========================================================================================")
print("STATIC MODEL EVALUATION ON G, H, J")
print("==========================================================================================")

for target in targets:
    folder = os.path.join("data", "AzSLD_Fingerspelling", target)
    all_files = sorted(glob.glob(os.path.join(folder, "*.*")))
    img_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    
    print(f"\n==================== EVALUATING TARGET: {target} (Found {len(img_files)} images) ====================")
    
    # Test up to 10 representative samples across the folder
    step = max(1, len(img_files) // 10)
    sample_files = img_files[::step][:10]
    
    for f in sample_files:
        fname = os.path.basename(f)
        img = imread_unicode(f)
        if img is None:
            continue
        res = landmarker.extract_landmarks(img)
        if res is None:
            print(f"File {fname}: No hand detected by Landmarker")
            continue
        raw_xyz, mirror_x = res
        coords = normalize_landmarks(raw_xyz, mirror_x=mirror_x)
        full84 = build_feature_vector_84(coords, velocity_xy=np.zeros(2, dtype=np.float64))
        
        # Level 1 forward
        l1_in = apply_scaler(full84, model_json["level1"]["scaler"])
        l1_cands = mlp_forward(model_json["level1"]["model"], l1_in)
        # cluster candidates: list of (cluster_id, prob)
        top_cluster = str(l1_cands[0][0])
        
        # Level 2 forward on top cluster
        c_entry = model_json["clusters"][top_cluster]
        f_indices = c_entry["featureIndices"]
        sub_in = full84 if len(f_indices) == len(full84) else full84[f_indices]
        l2_in = apply_scaler(sub_in, c_entry["scaler"])
        l2_cands = mlp_forward(c_entry["model"], l2_in)
        # l2_cands: sorted list of (letter, prob)
        
        # Also compute full 32-letter probability distribution across clusters
        # weighted by Level 1 cluster probability
        full_dist = {let: 0.0 for let in [
            'A', 'B', 'C', 'Ç', 'D', 'E', 'Ə', 'F', 'G', 'Ğ', 'H', 'X', 'I', 'İ',
            'J', 'K', 'Q', 'L', 'M', 'N', 'O', 'Ö', 'P', 'R', 'S', 'Ş', 'T', 'U',
            'Ü', 'V', 'Y', 'Z'
        ]}
        
        # Dispatched cluster distribution (how frontend currently evaluates it):
        dispatched_dist = {let: 0.0 for let in full_dist}
        for let, prob in l2_cands:
            dispatched_dist[let] = prob
            
        print(f"\nSample: {fname}")
        print(f"  Level 1 Clusters: {[(str(c), round(p, 4)) for c, p in l1_cands[:3]]}")
        print(f"  Dispatched to Cluster {top_cluster} (classes: {c_entry['letters']})")
        print(f"  Top 5 Predictions in Cluster {top_cluster}: {[(let, round(p, 4)) for let, p in l2_cands[:5]]}")
        print("  Specific Key Probabilities:")
        for k in key_letters:
            p_val = dispatched_dist.get(k, 0.0)
            print(f"    P({k}) = {p_val:.4f}", end="  ")
        print()


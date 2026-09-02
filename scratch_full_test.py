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

def full_test_class(letter):
    folder = os.path.join("data", "AzSLD_Fingerspelling", letter)
    files = sorted(glob.glob(os.path.join(folder, "*.*")))
    img_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    
    results = []
    l1_cluster_counts = {}
    
    for f in img_files:
        img = imread_unicode(f)
        if img is None: continue
        res = landmarker.extract_landmarks(img)
        if res is None: continue
        xyz, mirror_x = res
        coords = normalize_landmarks(xyz, mirror_x=mirror_x)
        full84 = build_feature_vector_84(coords, velocity_xy=np.zeros(2, dtype=np.float64))
        
        l1_in = apply_scaler(full84, model_json["level1"]["scaler"])
        l1_cands = mlp_forward(model_json["level1"]["model"], l1_in)
        top_cluster = str(l1_cands[0][0])
        l1_cluster_counts[top_cluster] = l1_cluster_counts.get(top_cluster, 0) + 1
        
        c_entry = model_json["clusters"][top_cluster]
        f_indices = c_entry["featureIndices"]
        sub_in = full84 if len(f_indices) == len(full84) else full84[f_indices]
        l2_in = apply_scaler(sub_in, c_entry["scaler"])
        l2_cands = mlp_forward(c_entry["model"], l2_in)
        
        results.append((os.path.basename(f), top_cluster, l2_cands[0][0], l2_cands[0][1], l1_cands))
        
    print(f"=== Letter {letter}: Tested {len(results)}/{len(img_files)} images ===")
    print(f"Level 1 Cluster distribution: {l1_cluster_counts}")
    pred_counts = {}
    for _, _, pred, _, _ in results:
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    print(f"Top predicted letters: {pred_counts}")
    
    # Show any misclassifications
    mis = [r for r in results if r[2] != letter]
    print(f"Misclassified: {len(mis)}/{len(results)}")
    for m in mis[:5]:
        print(f"  {m[0]}: dispatched to C{m[1]}, predicted {m[2]} ({m[3]:.2f})")
    print()

for l in ["G", "H", "J"]:
    full_test_class(l)


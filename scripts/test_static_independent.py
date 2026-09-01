#!/usr/bin/env python3
"""
Step 1: Test Static Hierarchical Model independently on H, P, C, K, Ç, M, Y.
Loads actual dataset images from data/AzSLD_Fingerspelling/<letter>,
extracts MediaPipe landmarks with LandmarkerWrapper (same as used in pipeline),
runs static classifier and logs:
- Actual letter
- Level-1 cluster predicted (and top cluster probabilities)
- Level-2 letter predictions (top 5 with confidences)
- Summary confusion matrix
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
    StaticHierarchicalModel,
    normalize_landmarks,
    build_feature_vector_84,
    apply_scaler,
    mlp_forward,
)

landmarker = LandmarkerWrapper("public/models/hand_landmarker.task", min_confidence=0.5)
static_model = StaticHierarchicalModel("public/models/azsl_hierarchical_model.json")

test_letters = ["H", "P", "C", "K", "Ç", "M", "Y"]

print("==========================================================================================")
print("INDEPENDENT STATIC CLASSIFIER EVALUATION ON H, P, C, K, Ç, M, Y")
print("==========================================================================================")

detailed_results = {let: [] for let in test_letters}

for let in test_letters:
    folder = os.path.join("data", "AzSLD_Fingerspelling", let)
    all_files = sorted(glob.glob(os.path.join(folder, "*.*")))
    # Filter image extensions
    img_files = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    print(f"\n--- Testing Letter: {let} (Found {len(img_files)} images in {folder}) ---")
    
    # Test up to 30 images
    test_sample = img_files[:30] if len(img_files) >= 30 else img_files
    
    for f in test_sample:
        fname = os.path.basename(f)
        img = cv2.imread(f)
        if img is None:
            continue
        res = landmarker.extract_landmarks(img)
        if res is None:
            continue
        raw_xyz, mirror_x = res
        
        # 1. Normalize landmarks
        coords = normalize_landmarks(raw_xyz, mirror_x=mirror_x)
        
        # 2. Build 84-dim feature vector (velocity = (0, 0))
        full84 = build_feature_vector_84(coords, velocity_xy=np.zeros(2, dtype=np.float64))
        
        # 3. Level-1 Cluster Dispatch
        l1_input = apply_scaler(full84, static_model.level1["scaler"])
        cluster_candidates = mlp_forward(static_model.level1["model"], l1_input)
        top_cluster = str(cluster_candidates[0][0])
        top_cluster_conf = cluster_candidates[0][1]
        
        # 4. Level-2 Letter Classification
        cluster_entry = static_model.clusters[top_cluster]
        feat_indices = cluster_entry["featureIndices"]
        sub_input = full84 if len(feat_indices) == len(full84) else full84[feat_indices]
        l2_input = apply_scaler(sub_input, cluster_entry["scaler"])
        letter_candidates = mlp_forward(cluster_entry["model"], l2_input)
        
        predicted_letter = letter_candidates[0][0]
        predicted_conf = letter_candidates[0][1]
        
        detailed_results[let].append({
            "file": fname,
            "top_cluster": top_cluster,
            "top_cluster_conf": top_cluster_conf,
            "pred_letter": predicted_letter,
            "pred_conf": predicted_conf,
            "top_candidates": letter_candidates[:5],
            "coords_sample": coords[0:3],
        })

print("\n==========================================================================================")
print("SUMMARY REPORT FOR INDEPENDENT STATIC CLASSIFIER")
print("==========================================================================================")
print("%-6s | %-6s | %-12s | %-20s | %-30s" % ("Letter", "Tested", "Accuracy", "Clusters Dispatched", "Top Predictions"))
print("-" * 84)

for let in test_letters:
    entries = detailed_results[let]
    n = len(entries)
    if n == 0:
        print(f"%-6s | %-6d | N/A" % (let, 0))
        continue
    correct = sum(1 for e in entries if e["pred_letter"] == let)
    acc = (correct / n) * 100
    
    # Clusters distribution
    clusters = [e["top_cluster"] for e in entries]
    c_counts = dict(zip(*np.unique(clusters, return_counts=True)))
    c_str = ", ".join([f"C{k}:{v}" for k, v in c_counts.items()])
    
    # Prediction distribution
    preds = [e["pred_letter"] for e in entries]
    p_counts = dict(zip(*np.unique(preds, return_counts=True)))
    p_str = ", ".join([f"{k}:{v}" for k, v in sorted(p_counts.items(), key=lambda x: x[1], reverse=True)[:3]])
    
    print("%-6s | %-6d | %5.1f%% (%2d/%2d) | %-20s | %-30s" % (let, n, acc, correct, n, c_str, p_str))

print("\n==========================================================================================")
print("EXAMINING SPECIFIC REPORTED CONFUSIONS:")
print("1. H -> P ?")
h_preds = [e["pred_letter"] for e in detailed_results["H"]]
print(f"   H predictions: {dict(zip(*np.unique(h_preds, return_counts=True)))}")

print("2. C -> K ?")
c_preds = [e["pred_letter"] for e in detailed_results["C"]]
print(f"   C predictions: {dict(zip(*np.unique(c_preds, return_counts=True)))}")

print("3. Ç -> K ?")
ch_preds = [e["pred_letter"] for e in detailed_results["Ç"]]
print(f"   Ç predictions: {dict(zip(*np.unique(ch_preds, return_counts=True)))}")

print("4. M predictions:")
m_preds = [e["pred_letter"] for e in detailed_results["M"]]
print(f"   M predictions: {dict(zip(*np.unique(m_preds, return_counts=True)))}")

print("5. Y predictions:")
y_preds = [e["pred_letter"] for e in detailed_results["Y"]]
print(f"   Y predictions: {dict(zip(*np.unique(y_preds, return_counts=True)))}")
print("==========================================================================================")


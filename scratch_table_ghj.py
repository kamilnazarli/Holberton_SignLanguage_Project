import sys, os, glob, json
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
    m = json.load(f)

targets = {
    "G": ["G_35604.jpg", "G_37083.jpg", "G_38454.jpg"],
    "H": ["H27.jpg", "ID--1093293880--__523447fa-bfb5-408a-88e2-59a0507e8891.jpg", "ID--1490799172--__f53a1dff-2503-49b8-ae46-88b23b9c4702.jpg"],
    "J": ["ID--1037386146--__db674a6e-8e0a-414b-9d82-ab3d5bd5bc8c.jpg", "ID--1401468589--__b996cb92-5c26-49bb-9a12-f0efe3e1c307.jpg", "J21.jpg"]
}

req_letters = ["G", "H", "J", "Ş", "P", "F", "Ç"]

print("======================================================================================================")
print("REPRESENTATIVE SAMPLE RUNS FOR G, H, J")
print("======================================================================================================")

for target, sample_names in targets.items():
    print(f"\n=================== TARGET: {target} ===================")
    for sname in sample_names:
        fpath = os.path.join("data", "AzSLD_Fingerspelling", target, sname)
        img = imread_unicode(fpath)
        if img is None:
            print(f"Error loading {fpath}")
            continue
        res = landmarker.extract_landmarks(img)
        if res is None:
            print(f"No hand in {sname}")
            continue
        xyz, mirror = res
        coords = normalize_landmarks(xyz, mirror_x=mirror)
        feat84 = build_feature_vector_84(coords, np.zeros(2))
        
        # Level 1
        l1_cands = mlp_forward(m["level1"]["model"], apply_scaler(feat84, m["level1"]["scaler"]))
        top_cluster = str(l1_cands[0][0])
        
        # Level 2
        c_entry = m["clusters"][top_cluster]
        f_idx = c_entry["featureIndices"]
        sub_feat = feat84 if len(f_idx) == len(feat84) else feat84[f_idx]
        l2_cands = mlp_forward(c_entry["model"], apply_scaler(sub_feat, c_entry["scaler"]))
        
        dispatched_dist = {let: 0.0 for let in req_letters}
        for let, prob in l2_cands:
            if let in dispatched_dist:
                dispatched_dist[let] = prob
                
        # Also compute end-to-end full 32-letter marginal distribution
        # P(letter) = sum_over_clusters [ P(Cluster c) * P(letter | Cluster c) ]
        full_marginal = {let: 0.0 for let in req_letters}
        for cid, c_prob in l1_cands:
            entry = m["clusters"][str(cid)]
            idx_list = entry["featureIndices"]
            s_feat = feat84 if len(idx_list) == len(feat84) else feat84[idx_list]
            sub_cands = mlp_forward(entry["model"], apply_scaler(s_feat, entry["scaler"]))
            for let, lprob in sub_cands:
                if let in full_marginal:
                    full_marginal[let] += c_prob * lprob
                    
        print(f"\nSample: {sname}")
        print(f"  Level 1 Dispatch: Cluster {top_cluster} (P={l1_cands[0][1]:.4f}), Runner-up Cluster {l1_cands[1][0]} (P={l1_cands[1][1]:.4f})")
        print(f"  Dispatched Cluster {top_cluster} Top 5: {[(let, round(prob, 4)) for let, prob in l2_cands[:5]]}")
        print(f"  Specific Letter Probabilities (Dispatched):")
        for k in req_letters:
            print(f"    P({k}) = {dispatched_dist[k]:.4f}")
        print(f"  Marginal Across All Clusters:")
        for k in req_letters:
            print(f"    P_marginal({k}) = {full_marginal[k]:.4f}")


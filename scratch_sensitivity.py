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

# Test H samples under slight wrist rotation or angle change
print("=== Sensitivity of H samples to Level 1 dispatch ===")
h_files = sorted(glob.glob("data/AzSLD_Fingerspelling/H/*.jpg"))
c4_count = 0
c6_count = 0
for f in h_files:
    img = imread_unicode(f)
    if img is None: continue
    res = landmarker.extract_landmarks(img)
    if res is None: continue
    xyz, mirror = res
    coords = normalize_landmarks(xyz, mirror_x=mirror)
    feat84 = build_feature_vector_84(coords, np.zeros(2))
    
    # Check Level 1 raw logits/probabilities
    l1_in = apply_scaler(feat84, m["level1"]["scaler"])
    l1_preds = mlp_forward(m["level1"]["model"], l1_in)
    p_c4 = next((p for c, p in l1_preds if str(c) == "4"), 0.0)
    p_c6 = next((p for c, p in l1_preds if str(c) == "6"), 0.0)
    
    # Test with slight tilt (+- 5 degrees or small landmark perturbation)
    # Rotating coords around Z axis slightly
    for angle in [-10, -5, 0, 5, 10]:
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rot_coords = coords.copy()
        # Rotate in XY plane
        x_new = rot_coords[:, 0] * cos_a - rot_coords[:, 1] * sin_a
        y_new = rot_coords[:, 0] * sin_a + rot_coords[:, 1] * cos_a
        rot_coords[:, 0] = x_new
        rot_coords[:, 1] = y_new
        rot_feat = build_feature_vector_84(rot_coords, np.zeros(2))
        rot_l1 = mlp_forward(m["level1"]["model"], apply_scaler(rot_feat, m["level1"]["scaler"]))
        top_c = str(rot_l1[0][0])
        if top_c == "4":
            c4_count += 1
        elif top_c == "6":
            c6_count += 1

print(f"H under slight rotation: Cluster 6 (correct cluster): {c6_count}, Cluster 4 (P cluster): {c4_count}")
print(f"Percentage routed to Cluster 4 (guaranteed P prediction): {c4_count / (c4_count + c6_count) * 100:.1f}%\n")

# Test J samples under slight rotation
print("=== Sensitivity of J samples to Level 1 dispatch ===")
j_files = sorted(glob.glob("data/AzSLD_Fingerspelling/J/*.jpg"))
c2_count = 0
c6_count = 0
other_count = 0
for f in j_files:
    img = imread_unicode(f)
    if img is None: continue
    res = landmarker.extract_landmarks(img)
    if res is None: continue
    xyz, mirror = res
    coords = normalize_landmarks(xyz, mirror_x=mirror)
    
    for angle in [-10, -5, 0, 5, 10]:
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rot_coords = coords.copy()
        x_new = rot_coords[:, 0] * cos_a - rot_coords[:, 1] * sin_a
        y_new = rot_coords[:, 0] * sin_a + rot_coords[:, 1] * cos_a
        rot_coords[:, 0] = x_new
        rot_coords[:, 1] = y_new
        rot_feat = build_feature_vector_84(rot_coords, np.zeros(2))
        rot_l1 = mlp_forward(m["level1"]["model"], apply_scaler(rot_feat, m["level1"]["scaler"]))
        top_c = str(rot_l1[0][0])
        if top_c == "2":
            c2_count += 1
        elif top_c == "6":
            c6_count += 1
        else:
            other_count += 1

print(f"J under slight rotation: Cluster 2 (J/C/Ç cluster): {c2_count}, Cluster 6 (F cluster): {c6_count}, Other: {other_count}")
print(f"Percentage routed to Cluster 6 (where F is): {c6_count / (c2_count + c6_count + other_count) * 100:.1f}%\n")


import sys, os, glob, json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from scripts.dynamic_dataset import LandmarkerWrapper
from scripts.static_model import (
    normalize_landmarks,
    build_feature_vector_84,
    apply_scaler,
    mlp_forward,
)
import cv2

def imread_unicode(path):
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

landmarker = LandmarkerWrapper("public/models/hand_landmarker.task", min_confidence=0.5)

with open("public/models/azsl_hierarchical_model.json", "r", encoding="utf-8") as f:
    m = json.load(f)

# Let's check how many H images predict Cluster 4 if we look at the raw logits of Level 1
h_files = sorted(glob.glob("data/AzSLD_Fingerspelling/H/*.jpg"))

print(f"{'File':<40} | {'C4 prob':<10} | {'C6 prob':<10} | {'Level 1 Top'}")
print("-" * 75)

for f in h_files:
    img = imread_unicode(f)
    if img is None: continue
    res = landmarker.extract_landmarks(img)
    if res is None: continue
    xyz, mirror = res
    coords = normalize_landmarks(xyz, mirror_x=mirror)
    feat84 = build_feature_vector_84(coords, np.zeros(2))
    l1 = mlp_forward(m["level1"]["model"], apply_scaler(feat84, m["level1"]["scaler"]))
    p_c4 = next((p for c, p in l1 if str(c) == "4"), 0.0)
    p_c6 = next((p for c, p in l1 if str(c) == "6"), 0.0)
    top_c = str(l1[0][0])
    print(f"{os.path.basename(f):<40} | {p_c4:<10.4f} | {p_c6:<10.4f} | Cluster {top_c}")


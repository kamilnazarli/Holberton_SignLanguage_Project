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

with open("public/models/azsl_hierarchical_model.json", "r", encoding="utf-8") as f:
    m = json.load(f)

# What are the Level 1 weights that separate Cluster 4 from Cluster 6?
# Level 1 has 84 inputs -> 48 hidden -> 24 hidden -> 6 outputs
# Classes: [1, 2, 3, 4, 5, 6]
# Let's feed P samples and H samples through Level 1 and print their exact cluster probabilities!

from scripts.dynamic_dataset import LandmarkerWrapper
import cv2

def imread_unicode(path):
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

landmarker = LandmarkerWrapper("public/models/hand_landmarker.task", min_confidence=0.5)

print("=== Evaluating H samples across all 6 Level 1 clusters ===")
for f in sorted(glob.glob("data/AzSLD_Fingerspelling/H/*.jpg"))[:5]:
    img = imread_unicode(f)
    if img is None: continue
    res = landmarker.extract_landmarks(img)
    if res is None: continue
    xyz, mirror = res
    coords = normalize_landmarks(xyz, mirror_x=mirror)
    feat84 = build_feature_vector_84(coords, np.zeros(2))
    l1 = mlp_forward(m["level1"]["model"], apply_scaler(feat84, m["level1"]["scaler"]))
    print(f"H ({os.path.basename(f)}): Level 1 = {[(str(c), round(p, 4)) for c, p in l1]}")

print("\n=== Evaluating P samples across all 6 Level 1 clusters ===")
for f in sorted(glob.glob("data/AzSLD_Fingerspelling/P/*.jpg"))[:5]:
    img = imread_unicode(f)
    if img is None: continue
    res = landmarker.extract_landmarks(img)
    if res is None: continue
    xyz, mirror = res
    coords = normalize_landmarks(xyz, mirror_x=mirror)
    feat84 = build_feature_vector_84(coords, np.zeros(2))
    l1 = mlp_forward(m["level1"]["model"], apply_scaler(feat84, m["level1"]["scaler"]))
    print(f"P ({os.path.basename(f)}): Level 1 = {[(str(c), round(p, 4)) for c, p in l1]}")


import os, glob
import cv2
import numpy as np
from scripts.dynamic_dataset import LandmarkerWrapper

landmarker = LandmarkerWrapper("public/models/hand_landmarker.task", min_confidence=0.5)

def imread_unicode(path):
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

sh_files = sorted(glob.glob("data/AzSLD_Fingerspelling/Ş/*.jpg"))
print(f"Total Ş images: {len(sh_files)}")
for f in sh_files[:10]:
    img = imread_unicode(f)
    if img is None: continue
    res = landmarker.extract_landmarks(img)
    fname = os.path.basename(f)
    if res is None:
        print(f"Ş ({fname}): No hand")
        continue
    raw_xyz, mirror = res
    tips = [4, 8, 12, 16, 20]
    names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    lines = []
    for idx, name in zip(tips, names):
        dist = np.linalg.norm(raw_xyz[idx] - raw_xyz[0])
        dy = raw_xyz[idx][1] - raw_xyz[0][1]
        lines.append(f"{name}: d={dist:.3f}, dy={dy:.3f}")
    print(f"Ş ({fname}): " + "; ".join(lines))


import os
import glob
import cv2
import numpy as np
from scripts.dynamic_dataset import LandmarkerWrapper

landmarker = LandmarkerWrapper("public/models/hand_landmarker.task", min_confidence=0.5)

def imread_unicode(path):
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def describe_hand(img_path):
    img = imread_unicode(img_path)
    if img is None:
        return "Cannot read"
    res = landmarker.extract_landmarks(img)
    if res is None:
        return "No hand"
    raw_xyz, mirror = res
    # 0 wrist, 4 thumb tip, 8 index tip, 12 middle tip, 16 ring tip, 20 pinky tip
    tips = [4, 8, 12, 16, 20]
    names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
    lines = []
    for idx, name in zip(tips, names):
        # distance from wrist (0)
        dist = np.linalg.norm(raw_xyz[idx] - raw_xyz[0])
        # y coordinate relative to wrist (negative means above wrist)
        dy = raw_xyz[idx][1] - raw_xyz[0][1]
        lines.append(f"{name}: dist_from_wrist={dist:.3f}, dy={dy:.3f}")
    return "; ".join(lines)

print("G_35604:", describe_hand("data/AzSLD_Fingerspelling/G/G_35604.jpg"))
print("G_37083:", describe_hand("data/AzSLD_Fingerspelling/G/G_37083.jpg"))
print("G_39821:", describe_hand("data/AzSLD_Fingerspelling/G/G_39821.jpg"))

sh_sample = sorted(glob.glob("data/AzSLD_Fingerspelling/Ş/*.jpg"))[0]
print(f"Ş ({os.path.basename(sh_sample)}):", describe_hand(sh_sample))

import sys, os, glob
import numpy as np
import cv2

sys.stdout.reconfigure(encoding="utf-8")

from scripts.dynamic_dataset import LandmarkerWrapper
from scripts.static_model import (
    normalize_landmarks,
    joint_angles_15,
    tip_distances_4,
    build_feature_vector_84,
    apply_scaler,
    mlp_forward,
)

def imread_unicode(path):
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

landmarker = LandmarkerWrapper("public/models/hand_landmarker.task", min_confidence=0.5)

def analyze_letter(letter):
    folder = os.path.join("data", "AzSLD_Fingerspelling", letter)
    files = sorted(glob.glob(os.path.join(folder, "*.*")))[:10]
    angles_list = []
    dists_list = []
    for f in files:
        img = imread_unicode(f)
        if img is None: continue
        res = landmarker.extract_landmarks(img)
        if res is None: continue
        xyz, mirror_x = res
        coords = normalize_landmarks(xyz, mirror_x=mirror_x)
        angles = joint_angles_15(coords)
        dists = tip_distances_4(coords)
        angles_list.append(angles)
        dists_list.append(dists)
        
    avg_a = np.mean(angles_list, axis=0)
    avg_d = np.mean(dists_list, axis=0)
    # Fingers: thumb(0..2), index(3..5), middle(6..8), ring(9..11), pinky(12..14)
    # flex angles: base_flex (0, 3, 6, 9, 12), tip_flex (1, 4, 7, 10, 13)
    print(f"Letter {letter}:")
    print(f"   Thumb  flex: base={avg_a[0]:.1f}°, tip={avg_a[1]:.1f}°")
    print(f"   Index  flex: base={avg_a[3]:.1f}°, tip={avg_a[4]:.1f}°")
    print(f"   Middle flex: base={avg_a[6]:.1f}°, tip={avg_a[7]:.1f}°")
    print(f"   Ring   flex: base={avg_a[9]:.1f}°, tip={avg_a[10]:.1f}°")
    print(f"   Pinky  flex: base={avg_a[12]:.1f}°, tip={avg_a[13]:.1f}°")
    print(f"   Tip gaps (T-I, I-M, M-R, R-P): {[round(float(x), 2) for x in avg_d]}")

for l in ["G", "Ş", "H", "P", "J", "Ç", "F"]:
    analyze_letter(l)


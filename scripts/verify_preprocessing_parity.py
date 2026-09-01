#!/usr/bin/env python3
"""
Step 2: Compare training preprocessing vs index.html JS preprocessing line-by-line.
Uses scripts.static_model which contains the exact training functions without importing TF.
"""

import sys, os, math
sys.path.insert(0, ".")
import numpy as np

from scripts.static_model import (
    LM as LM_TRAIN,
    FINGERS as FINGERS_TRAIN,
    TIP_PAIRS as TIP_PAIRS_TRAIN,
    normalize_landmarks as norm_train,
    joint_angles_15 as angles_train,
    tip_distances_4 as dists_train,
    build_feature_vector_84 as feat_train,
)

# Live JavaScript implementation from index.html replicated faithfully:
LM_JS = {
    "WRIST": 0,
    "THUMB_CMC": 1, "THUMB_MCP": 2, "THUMB_IP": 3, "THUMB_TIP": 4,
    "INDEX_MCP": 5, "INDEX_PIP": 6, "INDEX_TIP": 8,
    "MIDDLE_MCP": 9, "MIDDLE_PIP": 10, "MIDDLE_TIP": 12,
    "RING_MCP": 13, "RING_PIP": 14, "RING_TIP": 16,
    "PINKY_MCP": 17, "PINKY_PIP": 18, "PINKY_TIP": 20,
}

FINGERS_JS = [
    {"name": "thumb", "base": LM_JS["WRIST"], "j1": LM_JS["THUMB_MCP"], "j2": LM_JS["THUMB_IP"], "tip": LM_JS["THUMB_TIP"]},
    {"name": "index", "base": LM_JS["WRIST"], "j1": LM_JS["INDEX_MCP"], "j2": LM_JS["INDEX_PIP"], "tip": LM_JS["INDEX_TIP"]},
    {"name": "middle", "base": LM_JS["WRIST"], "j1": LM_JS["MIDDLE_MCP"], "j2": LM_JS["MIDDLE_PIP"], "tip": LM_JS["MIDDLE_TIP"]},
    {"name": "ring", "base": LM_JS["WRIST"], "j1": LM_JS["RING_MCP"], "j2": LM_JS["RING_PIP"], "tip": LM_JS["RING_TIP"]},
    {"name": "pinky", "base": LM_JS["WRIST"], "j1": LM_JS["PINKY_MCP"], "j2": LM_JS["PINKY_PIP"], "tip": LM_JS["PINKY_TIP"]},
]

TIP_PAIRS_JS = [
    [LM_JS["THUMB_TIP"], LM_JS["INDEX_TIP"]],
    [LM_JS["INDEX_TIP"], LM_JS["MIDDLE_TIP"]],
    [LM_JS["MIDDLE_TIP"], LM_JS["RING_TIP"]],
    [LM_JS["RING_TIP"], LM_JS["PINKY_TIP"]],
]

def vec_sub_js(a, b):
    return {"x": a["x"] - b["x"], "y": a["y"] - b["y"], "z": a["z"] - b["z"]}

def vec_mag_js(v):
    return math.sqrt(v["x"] * v["x"] + v["y"] * v["y"] + v["z"] * v["z"])

def vec_dist_js(a, b):
    return vec_mag_js(vec_sub_js(a, b))

def angle_between_js(v1, v2):
    mags = vec_mag_js(v1) * vec_mag_js(v2) or 1e-6
    cos = min(1.0, max(-1.0, (v1["x"] * v2["x"] + v1["y"] * v2["y"] + v1["z"] * v2["z"]) / mags))
    return math.acos(cos) * (180.0 / math.pi)

def normalize_landmarks_js(landmarks, mirror_x):
    wrist = landmarks[LM_JS["WRIST"]]
    shifted = [vec_sub_js(p, wrist) for p in landmarks]
    scale = vec_mag_js(shifted[LM_JS["MIDDLE_MCP"]])
    if scale < 1e-6:
        scale = 1e-6
    normalized = [{"x": p["x"] / scale, "y": p["y"] / scale, "z": p["z"] / scale} for p in shifted]
    if mirror_x:
        for p in normalized:
            p["x"] = -p["x"]
    return normalized

def joint_angles_js(coords):
    out = [0.0] * 15
    middle_dir = vec_sub_js(coords[LM_JS["MIDDLE_PIP"]], coords[LM_JS["MIDDLE_MCP"]])
    for i in range(len(FINGERS_JS)):
        f = FINGERS_JS[i]
        base_flex = angle_between_js(vec_sub_js(coords[f["base"]], coords[f["j1"]]), vec_sub_js(coords[f["j2"]], coords[f["j1"]]))
        tip_flex = angle_between_js(vec_sub_js(coords[f["j1"]], coords[f["j2"]]), vec_sub_js(coords[f["tip"]], coords[f["j2"]]))
        this_dir = vec_sub_js(coords[f["j2"]], coords[f["j1"]])
        spread = angle_between_js(this_dir, middle_dir)
        out[i * 3] = base_flex
        out[i * 3 + 1] = tip_flex
        out[i * 3 + 2] = spread
    return out

def tip_distances_js(coords):
    return [vec_dist_js(coords[p[0]], coords[p[1]]) for p in TIP_PAIRS_JS]

def build_feature_vector_js(coords, velocity):
    out = []
    for p in coords:
        out.extend([p["x"], p["y"], p["z"]])
    out.extend(joint_angles_js(coords))
    out.extend(tip_distances_js(coords))
    out.extend([velocity["x"], velocity["y"]])
    return np.array(out, dtype=np.float64)

# Create a random synthetic hand to test numerical equivalence
np.random.seed(42)
synthetic_raw = np.random.uniform(0.1, 0.9, (21, 3))
synthetic_raw_js = [{"x": float(synthetic_raw[i, 0]), "y": float(synthetic_raw[i, 1]), "z": float(synthetic_raw[i, 2])} for i in range(21)]

for mirror in [False, True]:
    v_train = feat_train(norm_train(synthetic_raw, mirror_x=mirror), velocity_xy=np.array([0.05, -0.02]))
    v_js = build_feature_vector_js(normalize_landmarks_js(synthetic_raw_js, mirror_x=mirror), velocity={"x": 0.05, "y": -0.02})
    
    max_err = np.max(np.abs(v_train - v_js))
    print(f"MirrorX={mirror}: Max absolute difference between Python and JS feature vectors = {max_err:.2e}")

print("==========================================================================================")
print("COMPARING FEATURE VECTOR LAYOUT:")
print(f"Python vector length: {len(v_train)}, JS vector length: {len(v_js)}")
print(f"Coords (63):  Py[0:5]={v_train[0:5]}  |  JS[0:5]={v_js[0:5]}")
print(f"Angles (15):  Py[63:68]={v_train[63:68]}  |  JS[63:68]={v_js[63:68]}")
print(f"Dists  (4):   Py[78:82]={v_train[78:82]}  |  JS[78:82]={v_js[78:82]}")
print(f"Veloc  (2):   Py[82:84]={v_train[82:84]}  |  JS[82:84]={v_js[82:84]}")


#!/usr/bin/env python3
"""
Static Letter Recognition Module for Azerbaijani Sign Language (AzSLD).

Preserves and evaluates the existing two-level hierarchical classifier model
stored in `public/models/azsl_hierarchical_model.json` without modifying
or retraining the underlying static weights.
"""

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# Landmark Indices
LM = {
    "WRIST": 0,
    "THUMB_CMC": 1, "THUMB_MCP": 2, "THUMB_IP": 3, "THUMB_TIP": 4,
    "INDEX_MCP": 5, "INDEX_PIP": 6, "INDEX_TIP": 8,
    "MIDDLE_MCP": 9, "MIDDLE_PIP": 10, "MIDDLE_TIP": 12,
    "RING_MCP": 13, "RING_PIP": 14, "RING_TIP": 16,
    "PINKY_MCP": 17, "PINKY_PIP": 18, "PINKY_TIP": 20,
}

FINGERS = [
    ("thumb", LM["WRIST"], LM["THUMB_MCP"], LM["THUMB_IP"], LM["THUMB_TIP"]),
    ("index", LM["WRIST"], LM["INDEX_MCP"], LM["INDEX_PIP"], LM["INDEX_TIP"]),
    ("middle", LM["WRIST"], LM["MIDDLE_MCP"], LM["MIDDLE_PIP"], LM["MIDDLE_TIP"]),
    ("ring", LM["WRIST"], LM["RING_MCP"], LM["RING_PIP"], LM["RING_TIP"]),
    ("pinky", LM["WRIST"], LM["PINKY_MCP"], LM["PINKY_PIP"], LM["PINKY_TIP"]),
]

TIP_PAIRS = [
    (LM["THUMB_TIP"], LM["INDEX_TIP"]),
    (LM["INDEX_TIP"], LM["MIDDLE_TIP"]),
    (LM["MIDDLE_TIP"], LM["RING_TIP"]),
    (LM["RING_TIP"], LM["PINKY_TIP"]),
]

LEVEL2_NARROW_INDICES = list(range(63, 82))
FULL_VECTOR_LENGTH = 84

AZ_ALPHABET = [
    "A", "B", "C", "Ç", "D", "E", "Ə", "F", "G", "Ğ", "H", "X", "I", "İ",
    "J", "K", "Q", "L", "M", "N", "O", "Ö", "P", "R", "S", "Ş", "T", "U",
    "Ü", "V", "Y", "Z",
]


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    mags = np.linalg.norm(v1) * np.linalg.norm(v2)
    if mags < 1e-9:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / mags, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def normalize_landmarks(landmarks_xyz: np.ndarray, mirror_x: bool = False) -> np.ndarray:
    """
    Wrist-origin, scale-normalized, canonical Right-hand mirroring.
    Matches normalizeLandmarks() in JS and extract_azsl_model.py exactly.
    """
    wrist = landmarks_xyz[LM["WRIST"]]
    shifted = landmarks_xyz - wrist
    scale = np.linalg.norm(shifted[LM["MIDDLE_MCP"]])
    if scale < 1e-6:
        scale = 1e-6
    normalized = shifted / scale
    if mirror_x:
        normalized = normalized.copy()
        normalized[:, 0] *= -1
    return normalized


def joint_angles_15(coords: np.ndarray) -> np.ndarray:
    """15 angles: for each of 5 fingers: base-flex, tip-flex, spread-vs-middle."""
    out = np.zeros(15, dtype=np.float64)
    middle_dir = coords[LM["MIDDLE_PIP"]] - coords[LM["MIDDLE_MCP"]]
    for i, (_, base, j1, j2, tip) in enumerate(FINGERS):
        base_flex = angle_between(coords[base] - coords[j1], coords[j2] - coords[j1])
        tip_flex = angle_between(coords[j1] - coords[j2], coords[tip] - coords[j2])
        this_dir = coords[j2] - coords[j1]
        spread = angle_between(this_dir, middle_dir)
        out[i * 3 : i * 3 + 3] = [base_flex, tip_flex, spread]
    return out


def tip_distances_4(coords: np.ndarray) -> np.ndarray:
    return np.array([np.linalg.norm(coords[a] - coords[b]) for a, b in TIP_PAIRS], dtype=np.float64)


def build_feature_vector_84(coords: np.ndarray, velocity_xy: Optional[np.ndarray] = None) -> np.ndarray:
    """84-dim: 63 coords + 15 angles + 4 tip distances + 2 velocity."""
    if velocity_xy is None:
        velocity_xy = np.zeros(2, dtype=np.float64)
    return np.concatenate([
        coords.flatten(),
        joint_angles_15(coords),
        tip_distances_4(coords),
        velocity_xy,
    ])


def apply_scaler(vec: np.ndarray, scaler_dict: Dict[str, List[float]]) -> np.ndarray:
    mean = np.array(scaler_dict["mean"], dtype=np.float64)
    std = np.array(scaler_dict["std"], dtype=np.float64)
    std = np.where(std > 1e-9, std, 1e-9)
    return (vec - mean) / std


def mlp_forward(model_dict: Dict[str, Any], input_vec: np.ndarray) -> List[Tuple[Any, float]]:
    """
    Replays trained MLPClassifier forward pass with ReLU and Softmax/Sigmoid.
    Returns list of (class_label, confidence) sorted descending.
    """
    activation = input_vec.astype(np.float64)
    layers = model_dict["layers"]
    num_layers = len(layers)

    for li, layer in enumerate(layers):
        w = np.array(layer["weights"], dtype=np.float64)
        b = np.array(layer["biases"], dtype=np.float64)
        out = np.dot(activation, w) + b
        if li < num_layers - 1:
            out = np.maximum(0.0, out)  # ReLU
        activation = out

    output_activation = model_dict.get("outputActivation", "softmax")
    classes = model_dict["classes"]

    if output_activation == "sigmoid_binary" or (len(classes) == 2 and activation.shape[-1] == 1):
        p1 = 1.0 / (1.0 + np.exp(-activation[0]))
        candidates = [
            (classes[0], float(1.0 - p1)),
            (classes[1], float(p1)),
        ]
    else:
        max_logit = np.max(activation)
        exps = np.exp(activation - max_logit)
        sum_exp = np.sum(exps) or 1e-9
        probs = exps / sum_exp
        candidates = [(classes[i], float(probs[i])) for i in range(len(classes))]

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


class StaticHierarchicalModel:
    """
    Evaluator for the two-level hierarchical AzSL classifier.
    Loads and runs the exact static model exported in public/models/azsl_hierarchical_model.json.
    """

    def __init__(self, model_path: str = "public/models/azsl_hierarchical_model.json"):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Static model JSON file not found: {model_path}")
        with open(model_path, "r", encoding="utf-8") as f:
            self.model_data = json.load(f)

        self.alphabet = self.model_data.get("alphabet", AZ_ALPHABET)
        self.level1 = self.model_data["level1"]
        self.clusters = self.model_data["clusters"]

    def predict_from_landmarks(
        self,
        raw_landmarks_xyz: np.ndarray,
        mirror_x: bool = False,
        velocity_xy: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Runs the two-level classifier on raw 21x3 landmarks.
        """
        coords = normalize_landmarks(raw_landmarks_xyz, mirror_x=mirror_x)
        if velocity_xy is None:
            velocity_xy = np.zeros(2, dtype=np.float64)
        if mirror_x:
            velocity_xy = np.array([-velocity_xy[0], velocity_xy[1]], dtype=np.float64)

        full84 = build_feature_vector_84(coords, velocity_xy)
        return self.predict_from_feature_vector(full84)

    def predict_from_feature_vector(self, full84: np.ndarray) -> Dict[str, Any]:
        """
        Runs Level-1 dispatch and Level-2 sub-classifier on 84-dim vector.
        """
        l1_scaled = apply_scaler(full84, self.level1["scaler"])
        cluster_cands = mlp_forward(self.level1["model"], l1_scaled)
        top_cluster = str(cluster_cands[0][0])

        cluster_entry = self.clusters.get(top_cluster)
        if cluster_entry is None:
            return {"label": None, "confidence": 0.0, "cluster": top_cluster, "candidates": []}

        feat_indices = cluster_entry.get("featureIndices", list(range(FULL_VECTOR_LENGTH)))
        if len(feat_indices) == len(full84):
            sub_input = full84
        else:
            sub_input = full84[feat_indices]

        l2_scaled = apply_scaler(sub_input, cluster_entry["scaler"])
        letter_cands = mlp_forward(cluster_entry["model"], l2_scaled)

        return {
            "label": letter_cands[0][0],
            "confidence": letter_cands[0][1],
            "cluster": int(top_cluster),
            "candidates": letter_cands[:3],
        }


# ============================================================================
# Landmark-Level Data Augmentation (Training-Only)
# ============================================================================
def rotate_landmarks(
    landmarks: np.ndarray,
    max_angles: Tuple[float, float, float] = (8.0, 8.0, 10.0),
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Applies small 3D rotation around wrist origin (0, 0, 0).
    max_angles: (max_x_deg, max_y_deg, max_z_deg) representing pitch, yaw, roll.
    """
    if rng is None:
        rng = np.random.RandomState()

    ax = np.radians(rng.uniform(-max_angles[0], max_angles[0]))
    ay = np.radians(rng.uniform(-max_angles[1], max_angles[1]))
    az = np.radians(rng.uniform(-max_angles[2], max_angles[2]))

    cx, sx = np.cos(ax), np.sin(ax)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)

    cy, sy = np.cos(ay), np.sin(ay)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)

    cz, sz = np.cos(az), np.sin(az)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    R = Rz @ Ry @ Rx
    return (landmarks @ R.T).astype(np.float64)


def scale_landmarks(
    landmarks: np.ndarray,
    scale_range: Tuple[float, float] = (0.92, 1.08),
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Slightly scales the entire hand around wrist/origin (0, 0, 0).
    """
    if rng is None:
        rng = np.random.RandomState()
    scale = rng.uniform(scale_range[0], scale_range[1])
    return (landmarks * scale).astype(np.float64)


def translate_landmarks(
    landmarks: np.ndarray,
    max_translation: float = 0.02,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Small displacement of the entire hand, simulating MediaPipe wrist localization noise.
    """
    if rng is None:
        rng = np.random.RandomState()
    if max_translation <= 0:
        return landmarks.copy()
    offset = rng.uniform(-max_translation, max_translation, size=(1, 3))
    return (landmarks + offset).astype(np.float64)


def jitter_landmarks(
    landmarks: np.ndarray,
    jitter_std: float = 0.008,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Adds small Gaussian noise to landmark coordinates to simulate tracking jitter.
    """
    if rng is None:
        rng = np.random.RandomState()
    if jitter_std <= 0:
        return landmarks.copy()
    noise = rng.normal(0.0, jitter_std, size=landmarks.shape)
    return (landmarks + noise).astype(np.float64)


def augment_landmarks(
    landmarks: np.ndarray,
    max_angles: Tuple[float, float, float] = (8.0, 8.0, 10.0),
    scale_range: Tuple[float, float] = (0.92, 1.08),
    max_translation: float = 0.02,
    jitter_std: float = 0.008,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Master augmentation pipeline applying rotation, scaling, jitter, and translation.
    Must be applied ONLY to training data, before build_feature_vector_84().
    """
    if rng is None:
        rng = np.random.RandomState()
    aug = landmarks.copy()
    if max_angles and any(a > 0 for a in max_angles):
        aug = rotate_landmarks(aug, max_angles=max_angles, rng=rng)
    if scale_range and (scale_range[0] != 1.0 or scale_range[1] != 1.0):
        aug = scale_landmarks(aug, scale_range=scale_range, rng=rng)
    if jitter_std > 0:
        aug = jitter_landmarks(aug, jitter_std=jitter_std, rng=rng)
    if max_translation > 0:
        aug = translate_landmarks(aug, max_translation=max_translation, rng=rng)
    return aug



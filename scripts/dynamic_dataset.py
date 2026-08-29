#!/usr/bin/env python3
"""
Dynamic Letter Dataset & Feature Extraction for Azerbaijani Sign Language (AzSLD).

Handles:
- MediaPipe Hand Landmark extraction (21 landmarks x (x,y,z) = 63 features per frame)
- Unicode path reading on Windows (imread_unicode)
- Temporal burst grouping from video frame numbers
- Missing landmark interpolation & forward-fill
- Temporal windowing, resampling, and padding (configurable sequence_length)
- Group/burst-aware train/val/test splitting to prevent data leakage
- Motion data augmentation (coordinate jitter, speed variation, frame dropping)
"""

import os
import re
import sys
import types
from typing import Dict, List, Optional, Tuple, Union

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Ensure doc_controls is mocked if tensorflow docs are missing
if "tensorflow" not in sys.modules:
    tf = types.ModuleType("tensorflow")
    tf_tools = types.ModuleType("tensorflow.tools")
    tf_docs = types.ModuleType("tensorflow.tools.docs")

    class _DocControls:
        @staticmethod
        def do_not_doc_inheritable(obj): return obj
        @staticmethod
        def do_not_generate_docs(obj): return obj

    tf_docs.doc_controls = _DocControls
    tf.tools = tf_tools
    tf_tools.docs = tf_docs
    sys.modules["tensorflow"] = tf
    sys.modules["tensorflow.tools"] = tf_tools
    sys.modules["tensorflow.tools.docs"] = tf_docs

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# The 7 dynamic classes specified for AzSLD
DYNAMIC_CLASSES = ["C", "D", "Ö", "Ş", "Ü", "Y", "Z"]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(DYNAMIC_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(DYNAMIC_CLASSES)}

# Folder aliases in AzSLD dataset (e.g. 'Ç' folder provides motion for 'C'/'Ç' class if needed)
FOLDER_CANDIDATES = {
    "C": ["C", "Ç"],
    "D": ["D"],
    "Ö": ["Ö"],
    "Ş": ["Ş", "S"],
    "Ü": ["Ü"],
    "Y": ["Y"],
    "Z": ["Z"],
}

LM_WRIST = 0
LM_MIDDLE_MCP = 9
NUM_LANDMARKS = 21
LANDMARK_DIM = 63  # 21 * 3

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SEQUENTIAL_NAME_RE = re.compile(r"^[A-Za-zÀ-ɏ]*_?(\d+)\.(jpg|jpeg|png|bmp|webp)$", re.IGNORECASE)


def imread_unicode(path: str) -> Optional[np.ndarray]:
    """Unicode-safe image reader for Windows OpenCV."""
    try:
        with open(path, "rb") as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def normalize_landmarks_63(landmarks_xyz: np.ndarray, mirror_x: bool = False) -> np.ndarray:
    """
    Normalizes (21, 3) landmarks:
    1. Translates wrist to (0, 0, 0)
    2. Scales by distance to Middle MCP
    3. Mirrors X if Left hand (canonical Right hand representation)
    Returns flattened (63,) array.
    """
    wrist = landmarks_xyz[LM_WRIST]
    shifted = landmarks_xyz - wrist
    scale = np.linalg.norm(shifted[LM_MIDDLE_MCP])
    if scale < 1e-6:
        scale = 1e-6
    normalized = shifted / scale
    if mirror_x:
        normalized = normalized.copy()
        normalized[:, 0] *= -1.0
    return normalized.flatten()


class LandmarkerWrapper:
    """Singleton-style wrapper around MediaPipe HandLandmarker."""

    def __init__(self, model_path: str = "public/models/hand_landmarker.task", min_confidence: float = 0.5):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Hand landmarker model not found at {model_path}")
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def extract_landmarks(self, img_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, bool]]:
        """
        Extracts raw (21, 3) landmarks and mirror_x boolean.
        Returns None if no hand detected.
        """
        if img_bgr is None:
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self.landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return None
        raw_xyz = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float64)
        mirror_x = bool(result.handedness) and result.handedness[0][0].category_name == "Left"
        return raw_xyz, mirror_x


def interpolate_missing_frames(frames: List[Optional[np.ndarray]]) -> List[np.ndarray]:
    """
    Fills None values in a sequence of (63,) vectors using linear interpolation
    or forward/backward fill.
    """
    n = len(frames)
    valid_indices = [i for i, f in enumerate(frames) if f is not None]
    if not valid_indices:
        return [np.zeros(LANDMARK_DIM, dtype=np.float32) for _ in range(n)]

    out = [f.copy() if f is not None else None for f in frames]

    first_valid = valid_indices[0]
    for i in range(first_valid):
        out[i] = out[first_valid].copy()

    last_valid = valid_indices[-1]
    for i in range(last_valid + 1, n):
        out[i] = out[last_valid].copy()

    for idx in range(len(valid_indices) - 1):
        start_idx = valid_indices[idx]
        end_idx = valid_indices[idx + 1]
        if end_idx - start_idx > 1:
            start_vec = out[start_idx]
            end_vec = out[end_idx]
            steps = end_idx - start_idx
            for k in range(1, steps):
                alpha = k / float(steps)
                out[start_idx + k] = (1.0 - alpha) * start_vec + alpha * end_vec

    return [o.astype(np.float32) for o in out]


def resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """
    Resamples a sequence of shape (L, D) to (target_length, D) via linear interpolation.
    """
    cur_len, feat_dim = sequence.shape
    if cur_len == target_length:
        return sequence.astype(np.float32)
    if cur_len == 1:
        return np.repeat(sequence, target_length, axis=0).astype(np.float32)

    old_indices = np.linspace(0, cur_len - 1, num=cur_len)
    new_indices = np.linspace(0, cur_len - 1, num=target_length)
    resampled = np.zeros((target_length, feat_dim), dtype=np.float32)

    for d in range(feat_dim):
        resampled[:, d] = np.interp(new_indices, old_indices, sequence[:, d])

    return resampled


def discover_bursts(folder_path: str, max_gap: int = 30) -> Tuple[List[List[str]], List[str]]:
    """
    Discovers sequential video bursts and standalone images in a folder.
    """
    if not os.path.isdir(folder_path):
        return [], []

    sequential = []
    standalone = []

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(IMAGE_EXTENSIONS):
            continue
        m = SEQUENTIAL_NAME_RE.match(fname)
        if m:
            sequential.append((int(m.group(1)), fname))
        else:
            standalone.append(fname)

    sequential.sort(key=lambda t: t[0])

    bursts = []
    curr_burst = []
    for fno, fname in sequential:
        if not curr_burst:
            curr_burst.append((fno, fname))
        else:
            prev_fno = curr_burst[-1][0]
            if 0 < fno - prev_fno <= max_gap:
                curr_burst.append((fno, fname))
            else:
                bursts.append([fn for _, fn in curr_burst])
                curr_burst = [(fno, fname)]
    if curr_burst:
        bursts.append([fn for _, fn in curr_burst])

    return bursts, standalone


class DynamicDatasetBuilder:
    """
    Builds the dataset of temporal sequences for the 7 dynamic AzSL classes.
    """

    def __init__(
        self,
        data_dir: str = "data/AzSLD_Fingerspelling",
        model_path: str = "public/models/hand_landmarker.task",
        sequence_length: int = 20,
        stride: int = 3,
        burst_gap_threshold: int = 30,
        min_detection_confidence: float = 0.5,
    ):
        self.data_dir = data_dir
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.stride = stride
        self.burst_gap_threshold = burst_gap_threshold
        self.landmarker = LandmarkerWrapper(model_path, min_detection_confidence)

    def extract_folder_features(self, folder_path: str) -> Dict[str, Optional[np.ndarray]]:
        """
        Extracts and caches (63,) normalized landmarks for all images in a folder.
        """
        features = {}
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(IMAGE_EXTENSIONS):
                continue
            full_path = os.path.join(folder_path, fname)
            img_bgr = imread_unicode(full_path)
            res = self.landmarker.extract_landmarks(img_bgr)
            if res is not None:
                raw_xyz, mirror_x = res
                features[fname] = normalize_landmarks_63(raw_xyz, mirror_x)
            else:
                features[fname] = None
        return features

    def build_sequences_for_class(
        self,
        cls_name: str,
        max_sequences: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, int, str]]:
        """
        Builds sequences for a single dynamic class.
        Returns list of (sequence_array_T_63, label_index, group_id).
        """
        label_idx = CLASS_TO_IDX[cls_name]
        candidate_folders = FOLDER_CANDIDATES.get(cls_name, [cls_name])

        sequences = []
        group_counter = 0

        for folder_name in candidate_folders:
            folder_path = os.path.join(self.data_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            feat_cache = self.extract_folder_features(folder_path)
            bursts, standalone = discover_bursts(folder_path, self.burst_gap_threshold)

            for burst in bursts:
                group_id = f"{cls_name}_burst_{group_counter}"
                group_counter += 1

                raw_burst_feats = [feat_cache.get(fn) for fn in burst]
                valid_feats = interpolate_missing_frames(raw_burst_feats)
                burst_len = len(valid_feats)

                if burst_len >= self.sequence_length:
                    for start in range(0, burst_len - self.sequence_length + 1, self.stride):
                        win = np.array(valid_feats[start : start + self.sequence_length], dtype=np.float32)
                        sequences.append((win, label_idx, group_id))
                    if (burst_len - self.sequence_length) % self.stride != 0:
                        tail_win = np.array(valid_feats[-self.sequence_length:], dtype=np.float32)
                        sequences.append((tail_win, label_idx, group_id))
                elif burst_len >= 3:
                    burst_mat = np.array(valid_feats, dtype=np.float32)
                    resampled = resample_sequence(burst_mat, self.sequence_length)
                    sequences.append((resampled, label_idx, group_id))

            if len(sequences) < 20 and standalone:
                for s_fn in standalone:
                    feat = feat_cache.get(s_fn)
                    if feat is not None:
                        group_id = f"{cls_name}_standalone_{group_counter}"
                        group_counter += 1
                        seq = np.repeat(feat[np.newaxis, :], self.sequence_length, axis=0).astype(np.float32)
                        sequences.append((seq, label_idx, group_id))

        if max_sequences and len(sequences) > max_sequences:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(sequences), size=max_sequences, replace=False)
            sequences = [sequences[i] for i in idx]

        return sequences

    def build_all(
        self,
        max_per_class: Optional[int] = 300,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Builds the entire dataset across all 7 dynamic classes.
        """
        X_all = []
        y_all = []
        groups_all = []

        print(f"Building dynamic dataset for classes: {DYNAMIC_CLASSES} (seq_len={self.sequence_length})...")
        for cls_name in DYNAMIC_CLASSES:
            seqs = self.build_sequences_for_class(cls_name, max_sequences=max_per_class)
            print(f"  Class '{cls_name}': generated {len(seqs)} sequences across unique groups")
            for seq, label_idx, grp in seqs:
                X_all.append(seq)
                y_all.append(label_idx)
                groups_all.append(grp)

        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.int64)
        return X, y, groups_all


def augment_sequence(
    sequence: np.ndarray,
    jitter_std: float = 0.012,
    drop_rate: float = 0.1,
    speed_range: Tuple[float, float] = (0.85, 1.15),
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Applies motion-preserving data augmentation:
    1. Spatial jitter: small Gaussian noise to (x, y, z) landmarks
    2. Random frame drop + linear interpolation
    3. Speed variation: slight temporal stretching/compression
    """
    if rng is None:
        rng = np.random.RandomState()

    seq = sequence.copy()
    seq_len, feat_dim = seq.shape

    if jitter_std > 0:
        noise = rng.normal(0.0, jitter_std, size=seq.shape).astype(np.float32)
        seq += noise

    if drop_rate > 0:
        mask = rng.rand(seq_len) > drop_rate
        if np.sum(mask) >= 3:
            kept_indices = np.where(mask)[0]
            new_seq = np.zeros_like(seq)
            for d in range(feat_dim):
                new_seq[:, d] = np.interp(np.arange(seq_len), kept_indices, seq[kept_indices, d])
            seq = new_seq

    speed = rng.uniform(speed_range[0], speed_range[1])
    if abs(speed - 1.0) > 0.03:
        new_len = max(4, int(seq_len * speed))
        stretched = resample_sequence(seq, new_len)
        seq = resample_sequence(stretched, seq_len)

    return seq.astype(np.float32)


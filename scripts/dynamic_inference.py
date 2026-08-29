#!/usr/bin/env python3
"""
Dynamic Gesture Inference Module for Azerbaijani Sign Language (AzSLD).

Provides continuous real-time and file-based dynamic letter classification:
- Rolling landmark buffer of configurable sequence length (default: 20 frames)
- Sliding-window inference with stride / step
- Missing landmark recovery and smoothing
- Temporal confidence debouncing and stability filtering
"""

import collections
import os
import sys
import time
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Ensure scripts directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.dynamic_dataset import (
    CLASS_TO_IDX,
    DYNAMIC_CLASSES,
    IDX_TO_CLASS,
    LANDMARK_DIM,
    LandmarkerWrapper,
    interpolate_missing_frames,
    normalize_landmarks_63,
)
from scripts.dynamic_model import DynamicGestureRecognizer


class DynamicSignPredictor:
    """
    Continuous streaming dynamic letter predictor.
    Maintains a rolling temporal queue of extracted hand landmarks.
    """

    def __init__(
        self,
        checkpoint_path: str = "models/dynamic_model.pt",
        model_asset_path: str = "public/models/hand_landmarker.task",
        sequence_length: int = 20,
        min_confidence: float = 0.60,
        stability_count: int = 3,
        device: Optional[str] = None,
    ):
        self.sequence_length = sequence_length
        self.min_confidence = min_confidence
        self.stability_count = stability_count

        # MediaPipe landmarker
        self.landmarker = LandmarkerWrapper(model_asset_path, min_confidence=0.5)

        # PyTorch Dynamic Model
        self.recognizer = DynamicGestureRecognizer(checkpoint_path=checkpoint_path, device=device)
        self.classes = self.recognizer.classes

        # Rolling buffers
        self.landmark_buffer: Deque[Optional[np.ndarray]] = collections.deque(maxlen=sequence_length)
        self.recent_predictions: Deque[Optional[str]] = collections.deque(maxlen=stability_count)
        self.consecutive_stable_label: Optional[str] = None
        self.consecutive_stable_count: int = 0

    def reset(self) -> None:
        """Resets temporal buffers."""
        self.landmark_buffer.clear()
        self.recent_predictions.clear()
        self.consecutive_stable_label = None
        self.consecutive_stable_count = 0

    def process_frame(
        self,
        img_bgr: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Processes a single camera / video frame:
        1. Detects hand and normalizes landmarks
        2. Appends to rolling temporal buffer
        3. If buffer is full, runs dynamic model
        4. Applies temporal stability filtering
        """
        res = self.landmarker.extract_landmarks(img_bgr)
        raw_landmarks = None
        mirror_x = False

        if res is not None:
            raw_xyz, mirror_x = res
            raw_landmarks = normalize_landmarks_63(raw_xyz, mirror_x=mirror_x)

        self.landmark_buffer.append(raw_landmarks)

        # Count detected frames in buffer
        valid_count = sum(1 for f in self.landmark_buffer if f is not None)

        if len(self.landmark_buffer) < self.sequence_length or valid_count < (self.sequence_length // 2):
            return {
                "label": None,
                "confidence": 0.0,
                "stable_label": None,
                "is_stable": False,
                "buffer_full": len(self.landmark_buffer) == self.sequence_length,
                "valid_frames_in_buffer": valid_count,
                "candidates": [],
                "raw_landmarks": raw_landmarks,
                "hand_detected": raw_landmarks is not None,
            }

        # Interpolate any missing frames in the temporal buffer
        buffered_list = list(self.landmark_buffer)
        interpolated = interpolate_missing_frames(buffered_list)
        seq_mat = np.array(interpolated, dtype=np.float32)

        # Run dynamic sequence model
        pred = self.recognizer.predict_sequence(seq_mat)
        raw_label = pred["label"]
        confidence = pred["confidence"]

        accepted_label = raw_label if confidence >= self.min_confidence else None

        # Temporal stability check
        if accepted_label is not None and accepted_label == self.consecutive_stable_label:
            self.consecutive_stable_count += 1
        else:
            self.consecutive_stable_label = accepted_label
            self.consecutive_stable_count = 1 if accepted_label is not None else 0

        is_stable = self.consecutive_stable_count >= self.stability_count

        return {
            "label": accepted_label,
            "confidence": confidence,
            "stable_label": self.consecutive_stable_label if is_stable else None,
            "is_stable": is_stable,
            "buffer_full": True,
            "valid_frames_in_buffer": valid_count,
            "candidates": pred["candidates"],
            "probabilities": pred["probabilities"],
            "raw_landmarks": raw_landmarks,
            "hand_detected": raw_landmarks is not None,
        }

    def predict_from_landmark_sequence(
        self,
        landmarks_sequence: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Direct inference on a pre-extracted sequence of shape (T, 63).
        """
        return self.recognizer.predict_sequence(landmarks_sequence)


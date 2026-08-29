#!/usr/bin/env python3
"""
Integrated Dual-Pipeline Azerbaijani Sign Language (AzSLD) Recognition System.

Integrates the existing static letter classifier with the new dynamic sequence model:
- Configurable modes: "static", "dynamic", and "auto"
- In "auto" mode:
  * Analyzes hand landmark motion trajectory over a temporal rolling buffer
  * Motion gating: prioritizes dynamic sequence model when active motion is detected
  * Stationary gating: prioritizes static classifier when hand is held steady
  * Temporal debounce / hold stability to prevent intermediate dynamic frames
    from falsely triggering static letters
"""

import collections
import os
import sys
import time
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

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
from scripts.static_model import StaticHierarchicalModel, normalize_landmarks


class IntegratedSignRecognizer:
    """
    Unified Azerbaijani Sign Language Recognition System.
    """

    def __init__(
        self,
        static_model_path: str = "public/models/azsl_hierarchical_model.json",
        dynamic_model_path: str = "models/dynamic_model.pt",
        model_asset_path: str = "public/models/hand_landmarker.task",
        mode: str = "auto",  # "static", "dynamic", or "auto"
        sequence_length: int = 20,
        motion_threshold: float = 0.045,
        static_confidence_threshold: float = 0.60,
        dynamic_confidence_threshold: float = 0.60,
        stability_frames: int = 4,
        device: Optional[str] = None,
    ):
        self.mode = mode.lower()
        self.sequence_length = sequence_length
        self.motion_threshold = motion_threshold
        self.static_confidence_threshold = static_confidence_threshold
        self.dynamic_confidence_threshold = dynamic_confidence_threshold
        self.stability_frames = stability_frames

        # 1. MediaPipe Hand Landmarker
        self.landmarker = LandmarkerWrapper(model_asset_path, min_confidence=0.5)

        # 2. Existing Static Model (Unmodified)
        self.static_model = StaticHierarchicalModel(static_model_path)

        # 3. New Dynamic Model
        self.dynamic_model: Optional[DynamicGestureRecognizer] = None
        if os.path.isfile(dynamic_model_path):
            self.dynamic_model = DynamicGestureRecognizer(checkpoint_path=dynamic_model_path, device=device)
        else:
            print(f"[WARN] Dynamic model checkpoint not found at {dynamic_model_path}. Dynamic mode disabled until trained.")

        # Temporal Buffers
        self.raw_landmarks_buffer: Deque[Optional[np.ndarray]] = collections.deque(maxlen=sequence_length)
        self.norm63_buffer: Deque[Optional[np.ndarray]] = collections.deque(maxlen=sequence_length)
        self.wrist_pos_buffer: Deque[Tuple[float, float]] = collections.deque(maxlen=sequence_length)

        # Output stability state
        self.last_candidate_label: Optional[str] = None
        self.candidate_count: int = 0
        self.confirmed_label: Optional[str] = None

    def set_mode(self, mode: str) -> None:
        """Sets operating mode: 'static', 'dynamic', or 'auto'."""
        mode_clean = mode.lower()
        if mode_clean not in ["static", "dynamic", "auto"]:
            raise ValueError(f"Invalid mode '{mode}'. Choose from 'static', 'dynamic', 'auto'.")
        self.mode = mode_clean
        self.reset()

    def reset(self) -> None:
        """Resets temporal buffers and confirmation state."""
        self.raw_landmarks_buffer.clear()
        self.norm63_buffer.clear()
        self.wrist_pos_buffer.clear()
        self.last_candidate_label = None
        self.candidate_count = 0
        self.confirmed_label = None

    def compute_motion_energy(self) -> float:
        """
        Computes motion energy (mean frame-to-frame displacement of landmarks)
        over the buffered sequence.
        """
        valid_frames = [f for f in self.norm63_buffer if f is not None]
        if len(valid_frames) < 3:
            return 0.0
        displacements = []
        for i in range(len(valid_frames) - 1):
            diff = np.linalg.norm(valid_frames[i + 1] - valid_frames[i])
            displacements.append(diff)
        return float(np.mean(displacements)) if displacements else 0.0

    def compute_wrist_velocity(self) -> Tuple[float, float]:
        """Computes current wrist velocity for static model feature vector."""
        if len(self.wrist_pos_buffer) < 2:
            return (0.0, 0.0)
        span = len(self.wrist_pos_buffer)
        oldest = self.wrist_pos_buffer[0]
        newest = self.wrist_pos_buffer[-1]
        vx = (newest[0] - oldest[0]) / span
        vy = (newest[1] - oldest[1]) / span
        return (float(vx), float(vy))

    def process_frame(
        self,
        img_bgr: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Processes a live camera / video frame:
        Returns:
          {
            "final_letter": "A",
            "confirmed_letter": "A" (when held for stability_frames),
            "source_pipeline": "static" | "dynamic",
            "confidence": 0.95,
            "motion_energy": 0.02,
            "is_dynamic_motion": False,
            "static_prediction": {...},
            "dynamic_prediction": {...},
            "hand_detected": True,
          }
        """
        res = self.landmarker.extract_landmarks(img_bgr)

        raw_xyz = None
        norm63 = None
        mirror_x = False

        if res is not None:
            raw_xyz, mirror_x = res
            norm63 = normalize_landmarks_63(raw_xyz, mirror_x=mirror_x)
            wrist_pt = (float(raw_xyz[0, 0]), float(raw_xyz[0, 1]))
            self.wrist_pos_buffer.append(wrist_pt)
        else:
            self.wrist_pos_buffer.clear()

        self.raw_landmarks_buffer.append(raw_xyz)
        self.norm63_buffer.append(norm63)

        if raw_xyz is None:
            self.reset()
            return {
                "final_letter": None,
                "confirmed_letter": None,
                "source_pipeline": None,
                "confidence": 0.0,
                "motion_energy": 0.0,
                "is_dynamic_motion": False,
                "static_prediction": None,
                "dynamic_prediction": None,
                "hand_detected": False,
            }

        # 1. Evaluate Static Model on current frame
        vx, vy = self.compute_wrist_velocity()
        static_res = self.static_model.predict_from_landmarks(
            raw_landmarks_xyz=raw_xyz,
            mirror_x=mirror_x,
            velocity_xy=np.array([vx, vy], dtype=np.float64),
        )

        # 2. Evaluate Dynamic Model on temporal sequence if available
        dynamic_res = None
        valid_count = sum(1 for f in self.norm63_buffer if f is not None)

        if self.dynamic_model is not None and len(self.norm63_buffer) == self.sequence_length and valid_count >= 8:
            interpolated = interpolate_missing_frames(list(self.norm63_buffer))
            seq_mat = np.array(interpolated, dtype=np.float32)
            dynamic_res = self.dynamic_model.predict_sequence(seq_mat)

        # 3. Decision Arbitration Logic
        motion_energy = self.compute_motion_energy()
        is_dynamic_motion = motion_energy >= self.motion_threshold

        chosen_letter = None
        chosen_confidence = 0.0
        chosen_source = None

        if self.mode == "static":
            if static_res["confidence"] >= self.static_confidence_threshold:
                chosen_letter = static_res["label"]
                chosen_confidence = static_res["confidence"]
                chosen_source = "static"

        elif self.mode == "dynamic":
            if dynamic_res and dynamic_res["confidence"] >= self.dynamic_confidence_threshold:
                chosen_letter = dynamic_res["label"]
                chosen_confidence = dynamic_res["confidence"]
                chosen_source = "dynamic"

        else:  # "auto" / "combined" mode
            dyn_conf = dynamic_res["confidence"] if dynamic_res else 0.0
            stat_conf = static_res["confidence"]

            if is_dynamic_motion and dyn_conf >= self.dynamic_confidence_threshold:
                # Active motion detected and dynamic model is confident -> prioritize dynamic sign
                chosen_letter = dynamic_res["label"]
                chosen_confidence = dyn_conf
                chosen_source = "dynamic"
            elif not is_dynamic_motion and stat_conf >= self.static_confidence_threshold:
                # Hand is held steady and static model is confident -> prioritize static sign
                chosen_letter = static_res["label"]
                chosen_confidence = stat_conf
                chosen_source = "static"
            elif dyn_conf > stat_conf and dyn_conf >= self.dynamic_confidence_threshold:
                chosen_letter = dynamic_res["label"]
                chosen_confidence = dyn_conf
                chosen_source = "dynamic"
            elif stat_conf >= self.static_confidence_threshold:
                chosen_letter = static_res["label"]
                chosen_confidence = stat_conf
                chosen_source = "static"

        # 4. Temporal Debounce / Stability Confirmation
        if chosen_letter is not None and chosen_letter == self.last_candidate_label:
            self.candidate_count += 1
        else:
            self.last_candidate_label = chosen_letter
            self.candidate_count = 1 if chosen_letter is not None else 0

        if self.candidate_count >= self.stability_frames:
            self.confirmed_label = self.last_candidate_label
        else:
            self.confirmed_label = None

        return {
            "final_letter": chosen_letter,
            "confirmed_letter": self.confirmed_label,
            "source_pipeline": chosen_source,
            "confidence": chosen_confidence,
            "motion_energy": motion_energy,
            "is_dynamic_motion": is_dynamic_motion,
            "mode": self.mode,
            "static_prediction": static_res,
            "dynamic_prediction": dynamic_res,
            "hand_detected": True,
        }


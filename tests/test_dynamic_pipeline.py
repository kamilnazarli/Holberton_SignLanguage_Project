#!/usr/bin/env python3
"""
Unit tests for the AzSL Dynamic Letter Recognition Pipeline.
Tests:
- 7 Dynamic classes: D, Ü, Y, Ö, Z, C, Ş
- Variable sequence lengths and padding/resampling
- Missing frame interpolation
- Data augmentation robustness
- PyTorch sequence model prediction
"""

import os
import sys
import unittest
import numpy as np
import torch

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.dynamic_dataset import (
    CLASS_TO_IDX,
    DYNAMIC_CLASSES,
    IDX_TO_CLASS,
    augment_sequence,
    interpolate_missing_frames,
    normalize_landmarks_63,
    resample_sequence,
)
from scripts.dynamic_model import (
    DynamicGestureModel,
    DynamicGestureRecognizer,
)


class TestDynamicPipeline(unittest.TestCase):

    def setUp(self):
        self.config = {
            "model_type": "gru",
            "input_dim": 63,
            "sequence_length": 20,
            "hidden_dim": 32,
            "num_layers": 1,
            "bidirectional": True,
            "dense_dim": 24,
            "dropout": 0.1,
            "num_classes": len(DYNAMIC_CLASSES),
            "classes": DYNAMIC_CLASSES,
        }
        self.recognizer = DynamicGestureRecognizer(config=self.config, device="cpu")

    def test_dynamic_classes_definition(self):
        expected_classes = ["C", "D", "Ö", "Ş", "Ü", "Y", "Z"]
        for c in expected_classes:
            self.assertIn(c, DYNAMIC_CLASSES)
        self.assertEqual(len(DYNAMIC_CLASSES), 7)
        self.assertEqual(len(CLASS_TO_IDX), 7)

    def test_landmark_normalization_shape(self):
        landmarks = np.random.randn(21, 3)
        norm63 = normalize_landmarks_63(landmarks, mirror_x=False)
        self.assertEqual(norm63.shape, (63,))
        self.assertAlmostEqual(norm63[0], 0.0)  # Wrist X is 0
        self.assertAlmostEqual(norm63[1], 0.0)  # Wrist Y is 0
        self.assertAlmostEqual(norm63[2], 0.0)  # Wrist Z is 0

    def test_missing_frame_interpolation(self):
        # Create a list with missing frames (Nones)
        v1 = np.ones(63, dtype=np.float32) * 1.0
        v4 = np.ones(63, dtype=np.float32) * 4.0
        frames = [v1, None, None, v4]

        interpolated = interpolate_missing_frames(frames)
        self.assertEqual(len(interpolated), 4)
        for vec in interpolated:
            self.assertIsNotNone(vec)
            self.assertEqual(vec.shape, (63,))

        # Intermediate values should be linearly interpolated: 2.0 and 3.0
        self.assertAlmostEqual(float(interpolated[1][0]), 2.0, places=4)
        self.assertAlmostEqual(float(interpolated[2][0]), 3.0, places=4)

    def test_resample_sequence(self):
        seq = np.random.randn(10, 63).astype(np.float32)
        resampled_20 = resample_sequence(seq, target_length=20)
        self.assertEqual(resampled_20.shape, (20, 63))

        resampled_5 = resample_sequence(seq, target_length=5)
        self.assertEqual(resampled_5.shape, (5, 63))

    def test_data_augmentation(self):
        seq = np.random.randn(20, 63).astype(np.float32)
        aug_seq = augment_sequence(seq, jitter_std=0.02, drop_rate=0.1, speed_range=(0.8, 1.2))
        self.assertEqual(aug_seq.shape, (20, 63))
        self.assertFalse(np.array_equal(seq, aug_seq))

    def test_model_prediction_output(self):
        seq = np.random.randn(20, 63).astype(np.float32)
        res = self.recognizer.predict_sequence(seq)

        self.assertIn("label", res)
        self.assertIn("confidence", res)
        self.assertIn("candidates", res)
        self.assertIn("probabilities", res)

        self.assertIn(res["label"], DYNAMIC_CLASSES)
        self.assertEqual(len(res["candidates"]), 7)
        self.assertEqual(len(res["probabilities"]), 7)

        # Sum of probabilities should be approximately 1.0
        prob_sum = sum(res["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, places=4)


if __name__ == "__main__":
    unittest.main()


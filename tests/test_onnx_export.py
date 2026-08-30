#!/usr/bin/env python3
"""
Unit tests for Dynamic ONNX Model Export and Feature Parity.
"""

import json
import os
import sys
import unittest

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from scripts.dynamic_dataset import (
    CLASS_TO_IDX,
    DYNAMIC_CLASSES,
    IDX_TO_CLASS,
    normalize_landmarks_63,
)
from scripts.dynamic_model import DynamicGestureRecognizer


class TestDynamicONNXExport(unittest.TestCase):
    def test_onnx_file_exists_and_size(self):
        onnx_path = "public/models/dynamic_model.onnx"
        self.assertTrue(os.path.isfile(onnx_path), f"ONNX file not found at {onnx_path}")
        self.assertGreater(os.path.getsize(onnx_path), 100_000, "ONNX model file size unexpectedly small")

    def test_label_order_exact_match(self):
        expected_classes = ["C", "D", "Ö", "Ş", "Ü", "Y", "Z"]
        self.assertEqual(DYNAMIC_CLASSES, expected_classes)
        for i, c in enumerate(expected_classes):
            self.assertEqual(CLASS_TO_IDX[c], i)
            self.assertEqual(IDX_TO_CLASS[i], c)

    def test_normalization_shape_and_range(self):
        coords = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            coords[i] = [0.5 + i * 0.01, 0.5 + i * 0.01, 0.0]

        vec = normalize_landmarks_63(coords, mirror_x=False)
        self.assertEqual(vec.shape, (63,))
        self.assertEqual(vec.dtype, np.float32)
        # Wrist at (0, 0, 0)
        self.assertAlmostEqual(float(vec[0]), 0.0, places=5)
        self.assertAlmostEqual(float(vec[1]), 0.0, places=5)
        self.assertAlmostEqual(float(vec[2]), 0.0, places=5)

    def test_pytorch_forward_shape(self):
        checkpoint_path = "models/dynamic_model.pt"
        if os.path.isfile(checkpoint_path):
            recognizer = DynamicGestureRecognizer(checkpoint_path=checkpoint_path, device="cpu")
            dummy_input = torch.randn(1, 20, 63)
            with torch.no_grad():
                logits = recognizer.model(dummy_input)
            self.assertEqual(logits.shape, (1, 7))


if __name__ == "__main__":
    unittest.main()


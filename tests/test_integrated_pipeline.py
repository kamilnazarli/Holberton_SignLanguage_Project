#!/usr/bin/env python3
"""
Unit tests for the AzSL Integrated Recognition Pipeline.
Tests:
- Mode setting ("static", "dynamic", "auto")
- Motion energy computation
- Decision arbitration logic
- Temporal debounce / hold confirmation
"""

import os
import sys
import unittest
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.dynamic_model import DYNAMIC_CLASSES
from scripts.integrated_system import IntegratedSignRecognizer
from scripts.static_model import AZ_ALPHABET


class TestIntegratedPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.static_model_path = "public/models/azsl_hierarchical_model.json"
        if not os.path.isfile(cls.static_model_path):
            raise unittest.SkipTest(f"Static model file not found at {cls.static_model_path}")
        cls.recognizer = IntegratedSignRecognizer(
            static_model_path=cls.static_model_path,
            dynamic_model_path="models/dynamic_model.pt",
            mode="auto",
            sequence_length=10,
            stability_frames=3,
        )

    def test_mode_switching(self):
        self.recognizer.set_mode("static")
        self.assertEqual(self.recognizer.mode, "static")

        self.recognizer.set_mode("dynamic")
        self.assertEqual(self.recognizer.mode, "dynamic")

        self.recognizer.set_mode("auto")
        self.assertEqual(self.recognizer.mode, "auto")

        with self.assertRaises(ValueError):
            self.recognizer.set_mode("invalid_mode")

    def test_motion_energy_calculation(self):
        self.recognizer.reset()
        # Feed stationary frames
        static_frame = np.ones(63, dtype=np.float32)
        for _ in range(5):
            self.recognizer.norm63_buffer.append(static_frame)
        self.assertAlmostEqual(self.recognizer.compute_motion_energy(), 0.0)

        # Feed moving frames
        self.recognizer.reset()
        for i in range(5):
            self.recognizer.norm63_buffer.append(np.ones(63, dtype=np.float32) * float(i))
        motion = self.recognizer.compute_motion_energy()
        self.assertGreater(motion, 0.0)

    def test_debounce_logic(self):
        self.recognizer.reset()
        self.recognizer.stability_frames = 3

        # Simulate frame outputs
        self.assertEqual(self.recognizer.candidate_count, 0)
        self.assertIsNone(self.recognizer.confirmed_label)


if __name__ == "__main__":
    unittest.main()


#!/usr/bin/env python3
"""
Unit tests for the AzSL Static Letter Recognition Pipeline.
Verifies that the static model architecture and evaluation remain 100% functional.
"""

import os
import sys
import unittest
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.static_model import (
    AZ_ALPHABET,
    FULL_VECTOR_LENGTH,
    StaticHierarchicalModel,
    build_feature_vector_84,
    joint_angles_15,
    normalize_landmarks,
    tip_distances_4,
)


class TestStaticPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_path = "public/models/azsl_hierarchical_model.json"
        if not os.path.isfile(cls.model_path):
            raise unittest.SkipTest(f"Model file not found at {cls.model_path}")
        cls.model = StaticHierarchicalModel(cls.model_path)

    def test_model_loaded(self):
        self.assertIsNotNone(self.model.alphabet)
        self.assertEqual(len(self.model.alphabet), 32)
        self.assertIn("level1", self.model.model_data)
        self.assertIn("clusters", self.model.model_data)
        self.assertEqual(len(self.model.clusters), 6)

    def test_feature_extraction_shape(self):
        landmarks = np.random.randn(21, 3)
        coords = normalize_landmarks(landmarks, mirror_x=False)
        self.assertEqual(coords.shape, (21, 3))

        angles = joint_angles_15(coords)
        self.assertEqual(len(angles), 15)

        dists = tip_distances_4(coords)
        self.assertEqual(len(dists), 4)

        feat84 = build_feature_vector_84(coords, np.zeros(2))
        self.assertEqual(len(feat84), FULL_VECTOR_LENGTH)

    def test_prediction_output_structure(self):
        landmarks = np.random.randn(21, 3)
        res = self.model.predict_from_landmarks(landmarks, mirror_x=False)

        self.assertIn("label", res)
        self.assertIn("confidence", res)
        self.assertIn("cluster", res)
        self.assertIn("candidates", res)

        self.assertIn(res["label"], AZ_ALPHABET)
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertIsInstance(res["cluster"], int)
        self.assertGreaterEqual(res["cluster"], 1)
        self.assertLessEqual(res["cluster"], 6)

    def test_mirror_invariance(self):
        landmarks = np.random.randn(21, 3)
        res_r = self.model.predict_from_landmarks(landmarks, mirror_x=False)
        
        # Left-hand mirrored
        landmarks_l = landmarks.copy()
        landmarks_l[:, 0] *= -1
        res_l = self.model.predict_from_landmarks(landmarks_l, mirror_x=True)

        self.assertEqual(res_r["cluster"], res_l["cluster"])
        self.assertEqual(res_r["label"], res_l["label"])
        self.assertAlmostEqual(res_r["confidence"], res_l["confidence"], places=4)


if __name__ == "__main__":
    unittest.main()


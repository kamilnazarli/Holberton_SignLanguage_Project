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

    def test_landmark_rotations(self):
        from scripts.static_model import rotate_landmarks
        landmarks = np.random.randn(21, 3)
        landmarks[0] = 0.0  # wrist at origin
        rng = np.random.RandomState(42)
        rotated = rotate_landmarks(landmarks, max_angles=(15.0, 15.0, 15.0), rng=rng)
        self.assertEqual(rotated.shape, (21, 3))
        np.testing.assert_allclose(rotated[0], [0.0, 0.0, 0.0], atol=1e-12)

    def test_landmark_scaling(self):
        from scripts.static_model import scale_landmarks
        landmarks = np.random.randn(21, 3)
        landmarks[0] = 0.0
        rng = np.random.RandomState(42)
        scaled = scale_landmarks(landmarks, scale_range=(0.9, 1.1), rng=rng)
        self.assertEqual(scaled.shape, (21, 3))
        np.testing.assert_allclose(scaled[0], [0.0, 0.0, 0.0], atol=1e-12)

    def test_landmark_translation_and_jitter(self):
        from scripts.static_model import translate_landmarks, jitter_landmarks
        landmarks = np.random.randn(21, 3)
        rng = np.random.RandomState(42)
        trans = translate_landmarks(landmarks, max_translation=0.03, rng=rng)
        self.assertEqual(trans.shape, (21, 3))

        jit = jitter_landmarks(landmarks, jitter_std=0.01, rng=rng)
        self.assertEqual(jit.shape, (21, 3))

    def test_augment_landmarks_pipeline(self):
        from scripts.static_model import augment_landmarks
        landmarks = np.random.randn(21, 3)
        landmarks[0] = 0.0
        rng1 = np.random.RandomState(123)
        aug1 = augment_landmarks(landmarks, rng=rng1)
        self.assertEqual(aug1.shape, (21, 3))

        # Test reproducibility with same seed
        rng2 = np.random.RandomState(123)
        aug2 = augment_landmarks(landmarks, rng=rng2)
        np.testing.assert_allclose(aug1, aug2)

        # Ensure valid 84-dimensional feature vector is built from augmented coords
        feat = build_feature_vector_84(aug1, np.zeros(2))
        self.assertEqual(len(feat), FULL_VECTOR_LENGTH)
        self.assertFalse(np.any(np.isnan(feat)))
        self.assertFalse(np.any(np.isinf(feat)))

    def test_soft_routing_k1_equals_hard_routing(self):
        landmarks = np.random.randn(21, 3)
        res_hard = self.model.predict_from_landmarks(landmarks, mode="hard")
        res_soft_k1 = self.model.predict_from_landmarks(landmarks, mode="soft", top_k=1)

        self.assertEqual(res_hard["label"], res_soft_k1["label"])
        self.assertEqual(res_hard["cluster"], res_soft_k1["cluster"])
        self.assertAlmostEqual(res_hard["confidence"], res_soft_k1["confidence"], places=5)
        self.assertEqual(res_soft_k1["evaluatedClusters"], 1)

    def test_soft_routing_multi_clusters(self):
        landmarks = np.random.randn(21, 3)
        res_k2 = self.model.predict_from_landmarks(landmarks, mode="soft", top_k=2)
        self.assertEqual(res_k2["evaluatedClusters"], 2)
        self.assertIn("distribution", res_k2)
        self.assertAlmostEqual(sum(res_k2["distribution"].values()), 1.0, places=5)

        res_k6 = self.model.predict_from_landmarks(landmarks, mode="soft", top_k=6)
        self.assertEqual(res_k6["evaluatedClusters"], 6)
        self.assertAlmostEqual(sum(res_k6["distribution"].values()), 1.0, places=5)
        # With K=6, all 32 alphabet letters must have probability mass
        self.assertEqual(len(res_k6["distribution"]), 32)


if __name__ == "__main__":
    unittest.main()




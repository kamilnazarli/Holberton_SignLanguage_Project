#!/usr/bin/env python3
"""
Deep Diagnostic Script for the 7 Dynamic Classes and Static Routing.
"""

import os
import sys
import glob
from typing import Dict, List

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scripts.dynamic_dataset import DYNAMIC_CLASSES, normalize_landmarks_63
from scripts.dynamic_model import DynamicGestureRecognizer
from scripts.static_model import StaticHierarchicalModel

def run():
    cache_path = "data/dynamic_landmarks_cache.npz"
    cache = np.load(cache_path, allow_pickle=True)
    X = cache["X"]  # (893, 20, 63)
    y = cache["y"]  # (893,)

    recognizer = DynamicGestureRecognizer("models/dynamic_model.pt", device="cpu")
    static_model = StaticHierarchicalModel("public/models/azsl_hierarchical_model.json")

    print("========================================================================================")
    print("TEST 1: WHAT DOES THE DYNAMIC GRU OUTPUT WHEN FED A COMPLETELY STATIC/FLAT HAND?")
    print("========================================================================================")
    # What if a stationary pose has slight jitter or enters dynamic mode?
    for idx, c in enumerate(DYNAMIC_CLASSES):
        indices = np.where(y == idx)[0]
        if len(indices) == 0:
            continue
        sample_seq = X[indices[0]]
        flat_seq = np.repeat(sample_seq[0:1, :], 20, axis=0)
        # Add tiny jitter 0.005
        jitter_seq = flat_seq + np.random.normal(0, 0.005, flat_seq.shape).astype(np.float32)
        res_flat = recognizer.predict_sequence(flat_seq)
        res_jitter = recognizer.predict_sequence(jitter_seq)
        print(f"Pose of {c:2s} repeated: -> Flat: {res_flat['label']:2s} ({res_flat['confidence']:.3f}) | Jittered: {res_jitter['label']:2s} ({res_jitter['confidence']:.3f})")

    print("\n========================================================================================")
    print("TEST 2: WHAT DOES THE DYNAMIC GRU OUTPUT ON STATIC LETTERS (A, B, M, S, etc.)?")
    print("========================================================================================")
    # Test real static letters from data/AzSLD_Fingerspelling
    test_static_letters = ["A", "B", "M", "S", "L", "O", "U", "K"]
    for let in test_static_letters:
        folder = os.path.join("data", "AzSLD_Fingerspelling", let)
        files = glob.glob(os.path.join(folder, "*.jpg"))
        if not files:
            continue
        # We don't have all raw landmark caches for static letters, but we can test static model predictions on M
        # and see if M in static model is confused with Y.

    print("\n========================================================================================")
    print("TEST 3: STATIC HIERARCHICAL CLASSIFIER - CLUSTERS AND M CONFUSION")
    print("========================================================================================")
    # Check which cluster M and Y belong to in the static model
    print(f"Static Model Clusters:")
    for cid, centry in static_model.clusters.items():
        letters = centry.get("letters", [])
        print(f"  Cluster {cid}: {letters}")
        if "M" in letters:
            print(f"    --> 'M' is in Cluster {cid}")
        if "Y" in letters:
            print(f"    --> 'Y' is in Cluster {cid}")

    print("\n========================================================================================")
    print("TEST 4: FULL EVALUATION OF GENUINE SEQUENCES PER DYNAMIC CLASS IN GRU")
    print("========================================================================================")
    for idx, c in enumerate(DYNAMIC_CLASSES):
        indices = np.where(y == idx)[0]
        preds = []
        confs = []
        confusions = {}
        for i in indices:
            res = recognizer.predict_sequence(X[i])
            preds.append(res["label"])
            confs.append(res["confidence"])
            if res["label"] != c:
                confusions[res["label"]] = confusions.get(res["label"], 0) + 1
        acc = np.mean([1 if p == c else 0 for p in preds]) * 100
        avg_conf = np.mean(confs)
        min_conf = np.min(confs)
        print(f"Class {c:2s} ({len(indices):3d} seqs): Acc = {acc:5.1f}% | Avg Conf = {avg_conf:.3f} | Min Conf = {min_conf:.3f} | Misclassifications: {confusions}")

    print("\n========================================================================================")
    print("TEST 5: ROLLING WINDOW ACCURACY DURING PARTIAL/EARLY STAGES OF GESTURES")
    print("========================================================================================")
    # What happens at frame 5, 10, 15, 20 of a gesture when padded with preceding resting frame?
    for idx, c in enumerate(DYNAMIC_CLASSES):
        indices = np.where(y == idx)[0]
        if len(indices) == 0:
            continue
        seq = X[indices[0]]  # (20, 63)
        rest_frame = seq[0:1] # (1, 63)
        
        # Test window at t=5 (15 rest frames + 5 motion frames)
        w5 = np.concatenate([np.repeat(rest_frame, 15, axis=0), seq[:5]], axis=0)
        # Test window at t=10 (10 rest frames + 10 motion frames)
        w10 = np.concatenate([np.repeat(rest_frame, 10, axis=0), seq[:10]], axis=0)
        # Test window at t=15 (5 rest frames + 15 motion frames)
        w15 = np.concatenate([np.repeat(rest_frame, 5, axis=0), seq[:15]], axis=0)
        # Test window at t=20 (full 20 motion frames)
        w20 = seq

        r5 = recognizer.predict_sequence(w5)
        r10 = recognizer.predict_sequence(w10)
        r15 = recognizer.predict_sequence(w15)
        r20 = recognizer.predict_sequence(w20)

        print(f"Class {c:2s} progression:")
        print(f"   t= 5 frames: {r5['label']:2s} ({r5['confidence']:.3f})")
        print(f"   t=10 frames: {r10['label']:2s} ({r10['confidence']:.3f})")
        print(f"   t=15 frames: {r15['label']:2s} ({r15['confidence']:.3f})")
        print(f"   t=20 frames: {r20['label']:2s} ({r20['confidence']:.3f})")

if __name__ == "__main__":
    run()


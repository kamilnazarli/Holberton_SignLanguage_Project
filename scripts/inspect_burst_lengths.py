#!/usr/bin/env python3
"""
Inspect raw burst lengths and frame counts for all 7 dynamic classes.
"""

import sys, os, glob, re
sys.path.insert(0, ".")
import numpy as np

from scripts.dynamic_dataset import get_dataset_inventory, DYNAMIC_CLASSES

inventory = get_dataset_inventory()
print("=== DATASET INVENTORY ===")
for c, info in inventory.items():
    print(f"Class {c:2s}: Discovered Bursts={len(info['bursts'])}, Usable={len(info['usable_bursts'])}, Windows={info['usable_windows']}")
    burst_lens = [b['length'] for b in info['usable_bursts']]
    if burst_lens:
        print(f"   Burst lengths: min={min(burst_lens)}, max={max(burst_lens)}, median={np.median(burst_lens):.1f}, mean={np.mean(burst_lens):.1f}")


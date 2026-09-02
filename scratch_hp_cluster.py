import sys, json
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from scripts.static_model import (
    apply_scaler,
    mlp_forward,
)

with open("public/models/azsl_hierarchical_model.json", "r", encoding="utf-8") as f:
    m = json.load(f)

# Let's inspect the Level 1 weights for Cluster 4 vs Cluster 6
c4_model = m["clusters"]["4"]["model"]
print("Cluster 4 classes:", c4_model["classes"]) # ['L', 'P']
print("Cluster 6 classes contains H?:", "H" in m["clusters"]["6"]["model"]["classes"])
print("Cluster 4 classes contains H?:", "H" in m["clusters"]["4"]["model"]["classes"])


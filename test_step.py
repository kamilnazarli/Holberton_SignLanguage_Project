import os
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.dynamic_model import DynamicGestureModel

m = DynamicGestureModel()
m.train()
x = torch.randn(32, 20, 63)
y = torch.randint(0, 7, (32,))

opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

print("1. Forward pass...", flush=True)
logits = m(x)
print(f"logits: {logits.shape}", flush=True)

print("2. Loss...", flush=True)
loss = crit(logits, y)
print(f"loss: {loss.item()}", flush=True)

print("3. Backward pass...", flush=True)
loss.backward()
print("Backward done!", flush=True)

print("4. Optimizer step...", flush=True)
opt.step()
print("Optimizer step done!", flush=True)


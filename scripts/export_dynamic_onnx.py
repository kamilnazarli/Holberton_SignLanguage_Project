#!/usr/bin/env python3
"""
Export trained AzSL Dynamic Letter PyTorch GRU Model to ONNX format.

Target:
- Input shape: [1, 20, 63]
- Output shape: [1, 7] (probabilities for 7 dynamic classes: C, D, Ö, Ş, Ü, Y, Z)
- Output files: public/models/dynamic_model.onnx and models/dynamic_model.onnx
"""

import os
import shutil
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.dynamic_model import (
    DYNAMIC_CLASSES,
    DynamicGestureModel,
    DynamicGestureRecognizer,
)


class ExportableDynamicModelWrapper(nn.Module):
    """Wraps DynamicGestureModel to output softmax probabilities for browser inference."""

    def __init__(self, core_model: DynamicGestureModel):
        super().__init__()
        self.core = core_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.core(x)
        probabilities = F.softmax(logits, dim=-1)
        return probabilities


def export_to_onnx(
    checkpoint_path: str = "models/dynamic_model.pt",
    output_path: str = "public/models/dynamic_model.onnx",
    seq_len: int = 20,
    feat_dim: int = 63,
) -> str:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading trained checkpoint from {checkpoint_path}...", flush=True)
    recognizer = DynamicGestureRecognizer(checkpoint_path=checkpoint_path, device="cpu")
    recognizer.model.eval()

    export_model = ExportableDynamicModelWrapper(recognizer.model)
    export_model.eval()

    dummy_input = torch.randn(1, seq_len, feat_dim, dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Exporting model to ONNX: {output_path} (input shape: [1, {seq_len}, {feat_dim}])...", flush=True)
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["landmark_sequence"],
        output_names=["probabilities"],
    )

    models_copy_path = "models/dynamic_model.onnx"
    os.makedirs(os.path.dirname(os.path.abspath(models_copy_path)), exist_ok=True)
    if os.path.abspath(output_path) != os.path.abspath(models_copy_path):
        shutil.copyfile(output_path, models_copy_path)

    print(f"[SUCCESS] ONNX model successfully exported to {output_path} ({os.path.getsize(output_path)} bytes)", flush=True)
    return output_path


if __name__ == "__main__":
    export_to_onnx(
        checkpoint_path="models/dynamic_model.pt",
        output_path="public/models/dynamic_model.onnx",
    )


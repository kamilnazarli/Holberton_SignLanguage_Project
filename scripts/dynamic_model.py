#!/usr/bin/env python3
"""
Dynamic Sequence Model for Azerbaijani Sign Language (AzSLD).

Implements PyTorch-based GRU and LSTM models for classifying the 7 dynamic
AzSL letters: D, Ü, Y, Ö, Z, C, Ş.

Features:
- Configurable sequence length, hidden dimension, layer count, and architecture (GRU / LSTM).
- Bidirectional support with temporal attention or last-step pooling.
- Deterministic label mapping for all 7 classes.
- Model checkpoint serialization, JSON config export, and real-time inference wrappers.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The 7 canonical dynamic classes
DYNAMIC_CLASSES = ["C", "D", "Ö", "Ş", "Ü", "Y", "Z"]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(DYNAMIC_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(DYNAMIC_CLASSES)}

DEFAULT_CONFIG = {
    "model_type": "gru",
    "input_dim": 63,
    "sequence_length": 20,
    "hidden_dim": 64,
    "num_layers": 2,
    "bidirectional": True,
    "dense_dim": 48,
    "dropout": 0.25,
    "num_classes": len(DYNAMIC_CLASSES),
    "classes": DYNAMIC_CLASSES,
}


class DynamicGestureModel(nn.Module):
    """
    Recurrent sequence model (GRU or LSTM) for dynamic sign language recognition.
    Input shape: (batch_size, sequence_length, 63)
    Output shape: (batch_size, 7)
    """

    def __init__(
        self,
        input_dim: int = 63,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 7,
        model_type: str = "gru",
        bidirectional: bool = True,
        dense_dim: int = 48,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.model_type = model_type.lower()
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.dense_dim = dense_dim
        self.dropout_rate = dropout

        # Input projection & layer norm
        self.input_norm = nn.LayerNorm(input_dim)

        # Recurrent backbone
        rnn_cls = nn.GRU if self.model_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        rnn_out_dim = hidden_dim * self.num_directions

        # Attention pooling layer
        self.attention = nn.Sequential(
            nn.Linear(rnn_out_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_dim, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (batch_size, seq_len, input_dim)
        Returns logits: (batch_size, num_classes)
        """
        x_norm = self.input_norm(x)
        rnn_out, _ = self.rnn(x_norm)  # (batch_size, seq_len, rnn_out_dim)

        # Attention-weighted temporal pooling
        attn_weights = F.softmax(self.attention(rnn_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(attn_weights * rnn_out, dim=1)  # (batch, rnn_out_dim)

        logits = self.classifier(context)  # (batch, num_classes)
        return logits

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Computes softmax probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.ndarray, list)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    return obj


class DynamicGestureRecognizer:
    """
    Inference and checkpoint management wrapper for the dynamic gesture model.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if checkpoint_path and os.path.isfile(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
        else:
            self.config = config or DEFAULT_CONFIG.copy()
            self.model = self._build_model(self.config).to(self.device)
            self.classes = self.config.get("classes", DYNAMIC_CLASSES)

        self.model.eval()

    def _build_model(self, cfg: Dict[str, Any]) -> DynamicGestureModel:
        return DynamicGestureModel(
            input_dim=cfg.get("input_dim", 63),
            hidden_dim=cfg.get("hidden_dim", 64),
            num_layers=cfg.get("num_layers", 2),
            num_classes=cfg.get("num_classes", len(DYNAMIC_CLASSES)),
            model_type=cfg.get("model_type", "gru"),
            bidirectional=cfg.get("bidirectional", True),
            dense_dim=cfg.get("dense_dim", 48),
            dropout=cfg.get("dropout", 0.25),
        )

    def save_checkpoint(self, path: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        checkpoint = {
            "config": self.config,
            "state_dict": self.model.state_dict(),
            "classes": self.classes,
            "extra_meta": to_serializable(extra_meta or {}),
        }
        torch.save(checkpoint, path)

        # Also write a matching config JSON
        json_path = os.path.splitext(path)[0] + "_config.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": to_serializable(self.config),
                "classes": self.classes,
                "extra_meta": to_serializable(extra_meta or {}),
            }, f, indent=2, ensure_ascii=False)

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.config = checkpoint["config"]
        self.classes = checkpoint.get("classes", DYNAMIC_CLASSES)
        self.model = self._build_model(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict_sequence(self, sequence: np.ndarray) -> Dict[str, Any]:
        """
        Predicts dynamic letter from a (sequence_length, 63) array.
        Returns:
          {
            "label": "D",
            "confidence": 0.94,
            "candidates": [("D", 0.94), ("Z", 0.03), ...],
            "probabilities": { "C": 0.01, "D": 0.94, ... }
          }
        """
        self.model.eval()
        if sequence.ndim == 2:
            x_tensor = torch.tensor(sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
        elif sequence.ndim == 3:
            x_tensor = torch.tensor(sequence, dtype=torch.float32, device=self.device)
        else:
            raise ValueError(f"Expected sequence of shape (T, 63) or (B, T, 63), got {sequence.shape}")

        probs = self.model.predict_probabilities(x_tensor).cpu().numpy()[0]

        candidates = [(self.classes[i], float(probs[i])) for i in range(len(self.classes))]
        candidates.sort(key=lambda t: t[1], reverse=True)

        prob_dict = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}

        return {
            "label": candidates[0][0],
            "confidence": candidates[0][1],
            "candidates": candidates,
            "probabilities": prob_dict,
        }


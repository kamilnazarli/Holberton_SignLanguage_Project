#!/usr/bin/env python3
"""
Training Pipeline for AzSL Dynamic Letter Recognition Model.

Trains a PyTorch sequence model on the 7 dynamic classes (D, Ü, Y, Ö, Z, C, Ş)
using MediaPipe Hand landmarks with class-stratified group-aware leak-free splitting.

Metrics computed:
- Top-1 Accuracy
- Macro F1, Weighted F1
- 7x7 Confusion Matrix
- Per-class Precision, Recall, F1, Support

Usage:
    python scripts/train_dynamic.py --epochs 60 --batch-size 32 --model-type gru
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

from scripts.dynamic_dataset import (
    CACHE_VERSION,
    CLASS_TO_IDX,
    DYNAMIC_CLASSES,
    IDX_TO_CLASS,
    DynamicDatasetBuilder,
    augment_sequence,
)
from scripts.dynamic_model import (
    DynamicGestureModel,
    DynamicGestureRecognizer,
    to_serializable,
)


class PurePythonAdamW:
    """Pure-Python AdamW optimizer compatible across all platforms and Python 3.13+."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
    ):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay > 0:
                p.mul_(1.0 - self.lr * self.weight_decay)
            self.m[i].mul_(b1).add_(grad, alpha=1.0 - b1)
            self.v[i].mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
            m_hat = self.m[i] / (1.0 - b1 ** self.t)
            v_hat = self.v[i] / (1.0 - b2 ** self.t)
            step_val = m_hat / (torch.sqrt(v_hat) + self.eps)
            p.sub_(self.lr * step_val)


def compute_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes cross entropy loss via log_softmax with optional class balancing weights."""
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs[torch.arange(len(targets), device=targets.device), targets]
    if weight is not None:
        sample_weights = weight[targets]
        return (nll * sample_weights).sum() / (sample_weights.sum() + 1e-8)
    return nll.mean()


class SignSequenceDataset(Dataset):
    """PyTorch Dataset for sign landmark sequences with optional on-the-fly augmentation."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False, seed: int = 42):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.X[idx]
        if self.augment:
            seq = augment_sequence(seq, jitter_std=0.012, drop_rate=0.1, speed_range=(0.85, 1.15), rng=self.rng)
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


def class_stratified_group_split(
    y: np.ndarray,
    groups: np.ndarray,
    classes: List[str],
    test_ratio: float = 0.20,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Dict[str, Tuple[int, int]]], List[str]]:
    """
    Performs a Class-Stratified Group-Aware Train/Val/Test Split.

    Guarantees:
    1. Group-aware: All windows originating from the same burst/group stay in exactly ONE split (zero leakage).
    2. Class-aware: Every dynamic class with >=3 groups has burst representation in Train, Val, and Test.
    3. Sparse class handling: Classes with <3 groups are allocated transparently without leakage and reported.

    Returns:
    - train_idx, val_idx, test_idx: numpy arrays of indices into dataset
    - breakdown: Dict mapping class -> {'train': (n_groups, n_samples), 'val': (...), 'test': (...)}
    - warnings: List of warning strings for sparse classes
    """
    rng = np.random.RandomState(seed)
    train_groups = set()
    val_groups = set()
    test_groups = set()

    breakdown = {}
    warnings = []

    for cls_idx, cls_name in enumerate(classes):
        cls_mask = (y == cls_idx)
        cls_groups = np.unique(groups[cls_mask])
        K = len(cls_groups)

        shuffled_groups = rng.permutation(cls_groups)

        if K >= 3:
            n_test = max(1, int(round(K * test_ratio)))
            n_val = max(1, int(round(K * val_ratio)))
            if n_test + n_val >= K:
                n_val = max(1, K - n_test - 1)

            c_test = set(shuffled_groups[:n_test])
            c_val = set(shuffled_groups[n_test : n_test + n_val])
            c_train = set(shuffled_groups[n_test + n_val :])
        elif K == 2:
            c_train = {shuffled_groups[0]}
            c_val = set()
            c_test = {shuffled_groups[1]}
            warnings.append(f"Class '{cls_name}' has only 2 discovered groups: 1 assigned to Train, 1 to Test (Val=0).")
        elif K == 1:
            c_train = {shuffled_groups[0]}
            c_val = set()
            c_test = set()
            warnings.append(f"Class '{cls_name}' has only 1 discovered group in source dataset: assigned to Train (Val=0, Test=0).")
        else:
            c_train, c_val, c_test = set(), set(), set()
            warnings.append(f"Class '{cls_name}' has 0 groups in dataset!")

        train_groups.update(c_train)
        val_groups.update(c_val)
        test_groups.update(c_test)

        n_tr_samples = int(sum(1 for g, cls in zip(groups, y) if cls == cls_idx and g in c_train))
        n_va_samples = int(sum(1 for g, cls in zip(groups, y) if cls == cls_idx and g in c_val))
        n_te_samples = int(sum(1 for g, cls in zip(groups, y) if cls == cls_idx and g in c_test))

        breakdown[cls_name] = {
            "train": (len(c_train), n_tr_samples),
            "val": (len(c_val), n_va_samples),
            "test": (len(c_test), n_te_samples),
        }

    train_idx = np.array([i for i, g in enumerate(groups) if g in train_groups], dtype=np.int64)
    val_idx = np.array([i for i, g in enumerate(groups) if g in val_groups], dtype=np.int64)
    test_idx = np.array([i for i, g in enumerate(groups) if g in test_groups], dtype=np.int64)

    return train_idx, val_idx, test_idx, breakdown, warnings


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: PurePythonAdamW,
    device: torch.device,
    class_weight: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = compute_cross_entropy_loss(logits, y_batch, weight=class_weight)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)

    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = compute_cross_entropy_loss(logits, y_batch)
        total_loss += loss.item() * len(y_batch)

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    total = max(1, len(all_targets))

    avg_loss = total_loss / total
    acc = float(np.mean(all_preds == all_targets)) if len(all_targets) > 0 else 0.0
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", labels=list(range(len(DYNAMIC_CLASSES))), zero_division=0))

    return avg_loss, acc, macro_f1, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description="Train AzSL dynamic sequence model with class-stratified group splitting")
    parser.add_argument("--data-dir", default="data/AzSLD_Fingerspelling")
    parser.add_argument("--model-asset", default="public/models/hand_landmarker.task")
    parser.add_argument("--cache-path", default="data/dynamic_landmarks_cache.npz")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--model-type", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuild feature cache from scratch")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("public/models", exist_ok=True)

    print("=========================================================================", flush=True)
    print("AzSL Dynamic Letter Recognition — Class-Stratified Group Training Pipeline", flush=True)
    print(f"Target classes:    {DYNAMIC_CLASSES} (7 dynamic letters)", flush=True)
    print(f"Sequence length:   {args.sequence_length} frames", flush=True)
    print(f"Architecture:      {args.model_type.upper()} (hidden={args.hidden_dim}, layers={args.num_layers}, bidirectional=True)", flush=True)
    print(f"Device:            {device}", flush=True)
    print("=========================================================================\n", flush=True)

    # Step 1: Load or extract dataset
    cache_valid = False
    if os.path.isfile(args.cache_path) and not args.rebuild_cache:
        try:
            cache_data = np.load(args.cache_path, allow_pickle=True)
            stored_version = str(cache_data.get("version", ""))
            if stored_version == CACHE_VERSION:
                X = cache_data["X"]
                y = cache_data["y"]
                groups = cache_data["groups"]
                print(f"[CACHE] Loaded {len(X)} sequences from existing cache ({args.cache_path}, version={stored_version}).", flush=True)
                cache_valid = True
            else:
                print(f"[CACHE] Cache version mismatch (found '{stored_version}', expected '{CACHE_VERSION}'). Rebuilding cache...", flush=True)
        except Exception as e:
            print(f"[CACHE] Failed to load cache: {e}. Rebuilding from scratch...", flush=True)

    if not cache_valid:
        print("[BUILD] Extracting landmarks and constructing temporal sequences from raw dataset...", flush=True)
        builder = DynamicDatasetBuilder(
            data_dir=args.data_dir,
            model_path=args.model_asset,
            sequence_length=args.sequence_length,
            stride=args.stride,
        )

        # Print source inventory
        inv = builder.get_dataset_inventory()
        print("\n--- Source Dataset Inventory ---", flush=True)
        header = "%-6s | %-12s | %-16s | %-16s | %-14s | %-18s" % (
            "Class", "Folders", "Discovered Bursts", "Standalone Images", "Usable Bursts", "Windowed Sequences"
        )
        print(header, flush=True)
        print("-" * 88, flush=True)
        for c, data in inv.items():
            print("%-6s | %-12s | %-16d | %-16d | %-14d | %-18d" % (
                c,
                ",".join(data["source_folders"]),
                data["discovered_bursts"],
                data["standalone_images"],
                data["usable_bursts"],
                data["sequences_after_windowing"],
            ), flush=True)
        print("-" * 88 + "\n", flush=True)

        X, y, groups_list = builder.build_all(max_per_class=350)
        groups = np.array(groups_list)
        os.makedirs(os.path.dirname(os.path.abspath(args.cache_path)), exist_ok=True)
        np.savez_compressed(args.cache_path, X=X, y=y, groups=groups, version=CACHE_VERSION)
        print(f"[BUILD] Saved new feature cache to {args.cache_path} (version={CACHE_VERSION})", flush=True)

    print(f"\nTotal Dataset: X={X.shape}, y={y.shape}, total_groups={len(np.unique(groups))}", flush=True)

    # Step 2: Class-Stratified Group-Aware Split
    train_idx, val_idx, test_idx, breakdown, split_warnings = class_stratified_group_split(
        y, groups, DYNAMIC_CLASSES, test_ratio=0.20, val_ratio=0.20, seed=args.seed
    )

    print("\n" + "=" * 88, flush=True)
    print("CLASS-STRATIFIED GROUP-AWARE SPLIT DISTRIBUTION TABLE")
    print("=" * 88, flush=True)
    table_header = "%-6s | %-24s | %-24s | %-24s" % (
        "Class", "Train (groups / samples)", "Val (groups / samples)", "Test (groups / samples)"
    )
    print(table_header, flush=True)
    print("-" * 88, flush=True)
    for cls_name in DYNAMIC_CLASSES:
        tr_g, tr_s = breakdown[cls_name]["train"]
        va_g, va_s = breakdown[cls_name]["val"]
        te_g, te_s = breakdown[cls_name]["test"]
        print("%-6s | %4d groups / %4d seqs   | %4d groups / %4d seqs   | %4d groups / %4d seqs" % (
            cls_name, tr_g, tr_s, va_g, va_s, te_g, te_s
        ), flush=True)
    print("-" * 88, flush=True)
    print(f"TOTALS | Train={len(train_idx)} sequences | Val={len(val_idx)} sequences | Test={len(test_idx)} sequences", flush=True)

    if split_warnings:
        print("\n--- Dataset Distribution Notes ---", flush=True)
        for w in split_warnings:
            print(f"[NOTE] {w}", flush=True)
    print("=" * 88 + "\n", flush=True)

    # Step 3: DataLoaders
    train_dataset = SignSequenceDataset(X[train_idx], y[train_idx], augment=(not args.no_augment), seed=args.seed)
    val_dataset = SignSequenceDataset(X[val_idx], y[val_idx], augment=False, seed=args.seed)
    test_dataset = SignSequenceDataset(X[test_idx], y[test_idx], augment=False, seed=args.seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Step 4: Model, Optimizer
    config = {
        "model_type": args.model_type,
        "input_dim": 63,
        "sequence_length": args.sequence_length,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "bidirectional": True,
        "dense_dim": 48,
        "dropout": 0.25,
        "num_classes": len(DYNAMIC_CLASSES),
        "classes": DYNAMIC_CLASSES,
    }

    model = DynamicGestureModel(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
        model_type=config["model_type"],
        bidirectional=config["bidirectional"],
        dense_dim=config["dense_dim"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = PurePythonAdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Class weights for loss function to balance sparse classes (C, Ş) with abundant classes
    tr_counts = np.bincount(y[train_idx], minlength=len(DYNAMIC_CLASSES))
    class_weights_np = len(train_idx) / (len(DYNAMIC_CLASSES) * np.maximum(tr_counts, 1).astype(np.float32))
    class_weights = torch.tensor(np.clip(class_weights_np, 0.1, 50.0), dtype=torch.float32, device=device)

    # Step 5: Training Loop
    print("Starting dynamic model training...", flush=True)
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        # Learning rate schedule: cosine decay
        decay_factor = 0.5 * (1.0 + np.cos(np.pi * epoch / args.epochs))
        optimizer.lr = max(1e-5, args.lr * decay_factor)

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, class_weight=class_weights)
        val_loss, val_acc, val_f1, _, _ = evaluate_model(model, val_loader, device)

        is_better = False
        if val_f1 > best_val_f1 + 1e-4:
            is_better = True
        elif abs(val_f1 - best_val_f1) <= 1e-4 and val_loss < best_val_loss:
            is_better = True

        if is_better:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
                  f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.1f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}% | Val Macro F1: {val_f1:.4f}", flush=True)

    print(f"\nTraining completed in {time.time() - t_start:.1f}s. Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}", flush=True)

    # Load best checkpoint
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Step 6: Full Test Evaluation on Stratified Test Set
    test_loss, test_acc, test_macro_f1, test_preds, test_targets = evaluate_model(
        model, test_loader, device
    )
    test_weighted_f1 = float(f1_score(test_targets, test_preds, average="weighted", labels=list(range(len(DYNAMIC_CLASSES))), zero_division=0))

    print("\n" + "=" * 88, flush=True)
    print("FINAL TEST EVALUATION RESULTS (LEAK-FREE CLASS-STRATIFIED TEST SET)", flush=True)
    print("=" * 88, flush=True)
    print(f"Test Accuracy:     {test_acc * 100:.2f}%", flush=True)
    print(f"Test Macro F1:     {test_macro_f1:.4f}", flush=True)
    print(f"Test Weighted F1:  {test_weighted_f1:.4f}", flush=True)
    print(f"Total Test Samples:{len(test_targets)}", flush=True)

    target_names = [DYNAMIC_CLASSES[i] for i in range(len(DYNAMIC_CLASSES))]
    all_labels = list(range(len(DYNAMIC_CLASSES)))
    cls_report = classification_report(
        test_targets,
        test_preds,
        labels=all_labels,
        target_names=target_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    print("\nClassification Report:", flush=True)
    print(classification_report(test_targets, test_preds, labels=all_labels, target_names=target_names, digits=4, zero_division=0), flush=True)

    cm = confusion_matrix(test_targets, test_preds, labels=all_labels)
    print("\n7x7 Confusion Matrix:", flush=True)
    header = "     " + " ".join([f"{c:>6}" for c in DYNAMIC_CLASSES])
    print(header, flush=True)
    for i, row in enumerate(cm):
        row_str = f"{DYNAMIC_CLASSES[i]:<4} " + " ".join([f"{val:>6d}" for val in row])
        print(row_str, flush=True)

    # Step 7: Save model checkpoints and metadata
    model_save_path = os.path.join(args.output_dir, "dynamic_model.pt")
    public_model_save_path = "public/models/dynamic_model.pt"

    eval_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cache_version": CACHE_VERSION,
        "config": config,
        "best_epoch": best_epoch,
        "metrics": {
            "test_accuracy": round(float(test_acc), 4),
            "test_macro_f1": round(float(test_macro_f1), 4),
            "test_weighted_f1": round(float(test_weighted_f1), 4),
            "validation_macro_f1": round(float(best_val_f1), 4),
        },
        "split_breakdown": breakdown,
        "split_warnings": split_warnings,
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist(),
        "classes": DYNAMIC_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
    }

    recognizer = DynamicGestureRecognizer(config=config, device="cpu")
    recognizer.model.load_state_dict(best_model_state)
    recognizer.save_checkpoint(model_save_path, extra_meta=eval_report)
    recognizer.save_checkpoint(public_model_save_path, extra_meta=eval_report)

    eval_report_path = os.path.join(args.output_dir, "dynamic_eval_report.json")
    with open(eval_report_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(eval_report), f, indent=2, ensure_ascii=False)

    label_map_path = os.path.join(args.output_dir, "dynamic_label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable({
            "classes": DYNAMIC_CLASSES,
            "class_to_idx": CLASS_TO_IDX,
            "idx_to_class": IDX_TO_CLASS,
        }), f, indent=2, ensure_ascii=False)

    print(f"\nModel checkpoint saved to: {model_save_path} and {public_model_save_path}", flush=True)
    print(f"Evaluation report saved to:  {eval_report_path}", flush=True)
    print(f"Label map saved to:          {label_map_path}", flush=True)


if __name__ == "__main__":
    main()

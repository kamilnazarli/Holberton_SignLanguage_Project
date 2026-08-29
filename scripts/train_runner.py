#!/usr/bin/env python3
import os
import sys
import traceback
import time
from datetime import datetime, timezone
import json

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from scripts.dynamic_dataset import (
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


class SignSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False, seed: int = 42):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        seq = self.X[idx]
        if self.augment:
            seq = augment_sequence(seq, jitter_std=0.012, drop_rate=0.1, speed_range=(0.85, 1.15), rng=self.rng)
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
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


def run_training(epochs: int = 50, batch_size: int = 32, seed: int = 42):
    log_path = "training_output.log"
    with open(log_path, "w", encoding="utf-8") as log_f:
        def log(msg):
            log_f.write(str(msg) + "\n")
            log_f.flush()
            print(msg, flush=True)

        try:
            log("=== Starting Training Run ===")
            torch.manual_seed(seed)
            np.random.seed(seed)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            log(f"Device: {device}")

            cache_path = "data/dynamic_landmarks_cache.npz"
            if os.path.isfile(cache_path):
                log(f"Loading cache from {cache_path}...")
                cache_data = np.load(cache_path, allow_pickle=True)
                X = cache_data["X"]
                y = cache_data["y"]
                groups = cache_data["groups"]
                log(f"Loaded {len(X)} sequences from cache.")
            else:
                log("Building dataset...")
                builder = DynamicDatasetBuilder(sequence_length=20, stride=3)
                X, y, groups_list = builder.build_all(max_per_class=350)
                groups = np.array(groups_list)
                np.savez_compressed(cache_path, X=X, y=y, groups=groups)

            # Leak-free splitting
            gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
            train_val_idx, test_idx = next(gss_test.split(X, y, groups=groups))
            train_val_groups = groups[train_val_idx]
            gss_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
            sub_train_idx, sub_val_idx = next(gss_val.split(X[train_val_idx], y[train_val_idx], groups=train_val_groups))

            train_idx = train_val_idx[sub_train_idx]
            val_idx = train_val_idx[sub_val_idx]
            log(f"Split sizes: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

            train_ds = SignSequenceDataset(X[train_idx], y[train_idx], augment=True, seed=seed)
            val_ds = SignSequenceDataset(X[val_idx], y[val_idx], augment=False, seed=seed)
            test_ds = SignSequenceDataset(X[test_idx], y[test_idx], augment=False, seed=seed)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

            config = {
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

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

            log(f"Training for {epochs} epochs...")
            best_val_f1 = -1.0
            best_model_state = None
            best_epoch = 0

            t0 = time.time()
            for epoch in range(1, epochs + 1):
                tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_loss, val_acc, val_f1, _, _ = evaluate_model(model, val_loader, criterion, device)
                scheduler.step()

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch = epoch
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
                    log(f"Epoch [{epoch:02d}/{epochs:02d}] "
                        f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.1f}% | "
                        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}% | Val Macro F1: {val_f1:.4f}")

            log(f"Training completed in {time.time() - t0:.1f}s. Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")

            if best_model_state is not None:
                model.load_state_dict(best_model_state)

            test_loss, test_acc, test_macro_f1, test_preds, test_targets = evaluate_model(
                model, test_loader, criterion, device
            )

            log("=== Final Test Results ===")
            log(f"Test Accuracy:  {test_acc * 100:.2f}%")
            log(f"Test Macro F1:  {test_macro_f1:.4f}")

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
            log("\nClassification Report:\n" + classification_report(test_targets, test_preds, labels=all_labels, target_names=target_names, digits=4, zero_division=0))

            cm = confusion_matrix(test_targets, test_preds, labels=all_labels)
            log("\n7x7 Confusion Matrix:")
            header = "     " + " ".join([f"{c:>6}" for c in DYNAMIC_CLASSES])
            log(header)
            for i, row in enumerate(cm):
                row_str = f"{DYNAMIC_CLASSES[i]:<4} " + " ".join([f"{val:>6d}" for val in row])
                log(row_str)

            os.makedirs("models", exist_ok=True)
            os.makedirs("public/models", exist_ok=True)

            eval_report = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config": config,
                "best_epoch": best_epoch,
                "metrics": {
                    "test_accuracy": round(float(test_acc), 4),
                    "test_macro_f1": round(float(test_macro_f1), 4),
                    "validation_macro_f1": round(float(best_val_f1), 4),
                },
                "classification_report": cls_report,
                "confusion_matrix": cm.tolist(),
                "classes": DYNAMIC_CLASSES,
                "class_to_idx": CLASS_TO_IDX,
            }

            recognizer = DynamicGestureRecognizer(config=config, device="cpu")
            recognizer.model.load_state_dict(best_model_state)
            recognizer.save_checkpoint("models/dynamic_model.pt", extra_meta=eval_report)
            recognizer.save_checkpoint("public/models/dynamic_model.pt", extra_meta=eval_report)

            with open("models/dynamic_eval_report.json", "w", encoding="utf-8") as f:
                json.dump(to_serializable(eval_report), f, indent=2, ensure_ascii=False)

            with open("models/dynamic_label_map.json", "w", encoding="utf-8") as f:
                json.dump(to_serializable({
                    "classes": DYNAMIC_CLASSES,
                    "class_to_idx": CLASS_TO_IDX,
                    "idx_to_class": IDX_TO_CLASS,
                }), f, indent=2, ensure_ascii=False)

            log("Model files and evaluation report saved successfully!")
        except Exception as e:
            log("EXCEPTION DURING TRAINING:")
            traceback.print_exc(file=log_f)
            traceback.print_exc()


if __name__ == "__main__":
    run_training(epochs=50, batch_size=32)


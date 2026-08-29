# Azerbaijani Sign Language (AzSLD) Dual Recognition System: Static & Dynamic Letters

This document describes the architecture, dataset construction, temporal modeling, training pipeline, and inference integration for both **static** and **dynamic** fingerspelling recognition in Azerbaijani Sign Language (AzSLD).

---

## 1. Static vs. Dynamic Letters in AzSL

The Azerbaijani sign language fingerspelling alphabet contains **32 letters**:

| Category | Count | Letters | Recognition Method |
| :--- | :--- | :--- | :--- |
| **Static Letters** | 25 | A, B, E, Ə, F, G, Ğ, H, X, I, İ, J, K, Q, L, M, N, O, P, R, S, T, U, V | Single-frame MediaPipe landmarks (84-dim geometric feature vector) $\rightarrow$ 2-level Hierarchical MLP |
| **Dynamic Letters** | 7 | **D, Ü, Y, Ö, Z, C, Ş** | Multi-frame temporal sequence of MediaPipe landmarks (20 frames $\times$ 63 features) $\rightarrow$ Bidirectional GRU Sequence Model |

---

## 2. Why Dynamic Letters Require Temporal Modeling

Static letters represent fixed handshapes held at a stationary point in space (e.g., 'A' is a fist with thumb alongside, 'B' is open palm with thumb across). A single image or camera snapshot provides sufficient geometric information to classify them.

Dynamic letters involve **movement over time**:
- **D**: Index finger traces a downwards arc / loop in the air.
- **Z**: Index finger traces a 'Z' zigzag stroke in the air.
- **Ö / Ü**: Hand signs the base vowel shape ('O' / 'U') while executing shaking / double-dot diacritic motion.
- **Y**: Hand executes a downward waving / sweeping motion.
- **C / Ş**: Base handshape combined with downward tail / strike diacritic motion.

Attempting to classify dynamic letters from a single static snapshot fails because:
1. Intermediate frames of dynamic gestures often resemble unrelated static handshapes.
2. The sign's identity is encoded in the **temporal trajectory (displacement, velocity, curvature)** rather than an instantaneous static pose.

---

## 3. Dataset Structure & Temporal Sequence Generation

### Genuine Temporal Bursts vs. Crowdsourced Photos
In the AzSLD dataset, video recordings were converted into sequentially numbered frames (e.g., `D_40234.jpg`, `Ü_100.jpg`, `C5.jpg`).

Our dataset pipeline (`scripts/dynamic_dataset.py`):
1. Identifies frame numbers in filenames via regex.
2. Groups contiguous frames into **temporal bursts** where frame gap $\le \text{threshold}$ (default: 30 frames).
3. Extracts sliding temporal windows of length $T = 20$ frames with stride $S = 3$ **strictly within each burst**.
4. Applies linear temporal interpolation and resampling for shorter bursts.
5. Employs **group-aware train/val/test splitting** (`GroupShuffleSplit`) based on burst IDs, preventing sliding windows from the same recording from appearing in both training and test sets.

```text
Video Burst: [Frame 101, Frame 103, Frame 104, Frame 106, ..., Frame 160]
                   │
                   ▼ (Sliding Window, T=20, Stride=3)
Window 1: [Frame 101 .. Frame 123]  ──┐
Window 2: [Frame 104 .. Frame 126]  ──┼──> All assigned to SAME split (Train or Test)
Window 3: [Frame 107 .. Frame 129]  ──┘
```

---

## 4. MediaPipe Landmark Extraction & Missing Frame Recovery

For each video frame:
1. Decode image safely via `imread_unicode` (handling Unicode characters such as `Ü`, `Ö`, `Ş`, `Ç` on Windows).
2. Detect 21 3D hand landmarks via MediaPipe Hand Landmarker.
3. Normalize coordinates:
   $$\mathbf{p}_{\text{norm}} = \frac{\mathbf{p} - \mathbf{p}_{\text{wrist}}}{\|\mathbf{p}_{\text{middle\_mcp}} - \mathbf{p}_{\text{wrist}}\|}$$
4. Canonical Right-hand mirroring: If a Left hand is detected, mirror $X$-coordinates ($x \leftarrow -x$).
5. Flatten 21 $(x, y, z)$ coordinates to a **63-dimensional feature vector** per frame.

### Missing Detection Recovery
If MediaPipe fails to detect a hand in intermediate frames (e.g., fast motion blur):
- **Linear interpolation**: Imputes coordinates linearly between the previous and next valid frames:
  $$\mathbf{p}(t) = (1 - \alpha)\mathbf{p}(t_{\text{prev}}) + \alpha \mathbf{p}(t_{\text{next}}), \quad \alpha = \frac{t - t_{\text{prev}}}{t_{\text{next}} - t_{\text{prev}}}$$
- **Edge padding**: Replicates the nearest valid landmark for leading/trailing missed frames.

---

## 5. Sequence Model Architecture

The dynamic letter model (`scripts/dynamic_model.py`) uses a recurrent neural network with attention-weighted temporal pooling:

```text
Input Landmark Sequence: (batch_size, 20, 63)
                   │
                   ▼
         Layer Normalization(63)
                   │
                   ▼
     2-Layer Bidirectional GRU (hidden_dim = 64)
                   │
         Shape: (batch_size, 20, 128)
                   │
                   ▼
    Temporal Attention Pooling:
    attn_weights = Softmax(Linear(128, 32) -> Tanh -> Linear(32, 1))
    context = sum(attn_weights * rnn_out, dim=1)
                   │
         Shape: (batch_size, 128)
                   │
                   ▼
         Dense Layer(128 -> 48) + ReLU + Dropout(0.25)
                   │
                   ▼
         Linear Projection(48 -> 7)
                   │
                   ▼
         Softmax Output: [C, D, Ö, Ş, Ü, Y, Z]
```

---

## 6. Integrated Dual-Pipeline Arbitration

The integrated system (`scripts/integrated_system.py`) provides three operating modes:

```text
Camera Stream
      │
      ▼
MediaPipe Hand Landmarker
      │
      ├───────────────────────────────┐
      ▼                               ▼
Static Classifier               Dynamic Buffer (T=20)
(Single Frame, 84-dim)                │
      │                               ▼
      │                         Dynamic Model
      │                         (Sequence, 20x63)
      │                               │
      ▼                               ▼
Static Probs [32 letters]       Dynamic Probs [7 letters]
      │                               │
      └───────────────┬───────────────┘
                      ▼
        Dual Arbitration Engine
        (Motion Gating + Temporal Debounce)
                      │
                      ▼
              Final Letter Output
```

### Operating Modes
1. **`mode = "static"`**: Evaluates only the preserved static hierarchical model.
2. **`mode = "dynamic"`**: Evaluates only the dynamic sequence model.
3. **`mode = "auto"`**: Intelligent motion gating and arbitration:
   - Calculates **motion energy** $E = \frac{1}{T-1}\sum_{t=1}^{T-1} \|\mathbf{p}_{t+1} - \mathbf{p}_t\|$.
   - If $E \ge \text{motion\_threshold}$ ($0.045$) and dynamic model confidence $\ge 0.60 \rightarrow$ dynamic letter selected.
   - If hand is stationary ($E < \text{motion\_threshold}$) and static model confidence $\ge 0.60 \rightarrow$ static letter selected.
   - **Temporal Debounce**: Requires a prediction to remain consistent for $k=4$ consecutive frames before committing, preventing intermediate frames of dynamic letters from false-firing static predictions.

---

## 7. Commands Reference

### Training the Dynamic Model
```powershell
python scripts/train_dynamic.py --epochs 60 --batch-size 32 --hidden-dim 64 --model-type gru
```

### Evaluating Static, Dynamic & Integrated Models
```powershell
python scripts/evaluate_models.py
```

### Running Unit Tests
```powershell
python -m unittest discover tests
```

---

## 8. Performance Benchmark Results

| Subsystem | Evaluation Metric | Result |
| :--- | :--- | :--- |
| **Static Classifier** | 5-Fold E2E Cross-Validation Accuracy (32 letters) | **89.66%** |
| **Dynamic Sequence Model** | Test Set Accuracy (7 classes) | **99.20%** |
| **Dynamic Sequence Model** | Validation Macro F1 | **0.9824** |
| **Dynamic Sequence Model** | Test Macro F1 | **0.8511** |
| **Inference Speed** | Frame Processing Latency | **~26 - 36 FPS (27.8 - 38.5 ms/frame)** |


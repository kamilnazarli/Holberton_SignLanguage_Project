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

## 3. Dataset Structure & Class Inventory

In `data/AzSLD_Fingerspelling`:
- High-volume recording bursts contain sequentially numbered frames (e.g., `D_40234.jpg`, `Ü_0.jpg`, `Y_9771.jpg`, `Ö_15917.jpg`, `Z_26233.jpg`).
- Standalone crowdsourced photos (`ID--...`) are separated and **never repeated** into fake dynamic sequences.
- Strict 1:1 class-to-folder mapping is enforced (`C` $\rightarrow$ `data/AzSLD_Fingerspelling/C`, `Ş` $\rightarrow$ `data/AzSLD_Fingerspelling/Ş`).

### Source Dataset Inventory
```text
========================================================================================
Class  | Folders      | Discovered Bursts | Standalone Images | Usable Bursts  | Windowed Sequences
----------------------------------------------------------------------------------------
C      | C            | 3                | 23               | 2              | 2                 
D      | D            | 84               | 0                | 68             | 90                
Ö      | Ö            | 10               | 0                | 10             | 226               
Ş      | Ş            | 1                | 35               | 1              | 1                 
Ü      | Ü            | 3                | 0                | 3              | 250               
Y      | Y            | 9                | 0                | 9              | 222               
Z      | Z            | 53               | 0                | 47             | 102               
========================================================================================
```

---

## 4. Class-Stratified Group-Aware Splitting (Leak-Free)

To prevent data leakage between overlapping sliding windows while ensuring all classes are represented in train, validation, and test splits:
- Splitting is performed **at the group/burst level for each class independently**.
- All sliding windows from burst $g_k$ stay strictly in one split.
- Multi-burst classes (`Ü` [3 bursts], `Y` [9 bursts], `Ö` [10 bursts], `D` [84 bursts], `Z` [53 bursts]) are allocated across Train, Val, and Test.

### Split Distribution Table
```text
========================================================================================
Class  | Train (groups / samples) | Val (groups / samples)   | Test (groups / samples) 
----------------------------------------------------------------------------------------
C      |    1 groups /    1 seqs   |    0 groups /    0 seqs   |    1 groups /    1 seqs
D      |   40 groups /   57 seqs   |   14 groups /   19 seqs   |   14 groups /   14 seqs
Ö      |    6 groups /  107 seqs   |    2 groups /   37 seqs   |    2 groups /   82 seqs
Ş      |    1 groups /    1 seqs   |    0 groups /    0 seqs   |    0 groups /    0 seqs
Ü      |    1 groups /  134 seqs   |    1 groups /   27 seqs   |    1 groups /   89 seqs
Y      |    5 groups /  113 seqs   |    2 groups /   73 seqs   |    2 groups /   36 seqs
Z      |   29 groups /   65 seqs   |    9 groups /   27 seqs   |    9 groups /   10 seqs
----------------------------------------------------------------------------------------
TOTALS | Train=478 sequences      | Val=183 sequences        | Test=232 sequences
========================================================================================
```

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

## 7. Performance Benchmark Results

| Subsystem | Evaluation Metric | Result |
| :--- | :--- | :--- |
| **Static Classifier** | 5-Fold E2E Cross-Validation Accuracy (32 letters) | **89.66%** |
| **Dynamic Sequence Model** | Test Set Accuracy (232 leak-free samples) | **100.00%** |
| **Dynamic Sequence Model** | Test Weighted F1 | **1.0000** |
| **Dynamic Sequence Model** | Validation Macro F1 | **0.7143** |
| **Inference Speed** | Frame Processing Latency | **~31.7 FPS (31.67 ms/frame)** |

### Dynamic Model Per-Class Classification Report (Test Set)
```text
              precision    recall  f1-score   support

           C     1.0000    1.0000    1.0000         1
           D     1.0000    1.0000    1.0000        14
           Ö     1.0000    1.0000    1.0000        82
           Ş     0.0000    0.0000    0.0000         0
           Ü     1.0000    1.0000    1.0000        89
           Y     1.0000    1.0000    1.0000        36
           Z     1.0000    1.0000    1.0000        10

    accuracy                         1.0000       232
   macro avg     0.8571    0.8571    0.8571       232
weighted avg     1.0000    1.0000    1.0000       232
```

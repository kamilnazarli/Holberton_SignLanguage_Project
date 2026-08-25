#!/usr/bin/env python3
"""
Build the two-level hierarchical AzSL fingerspelling classifier.

IMPORTANT — provenance note (read before trusting any number below):
This pipeline is *inspired by* the general "clustered/hybrid model" idea
described in Hasanov et al. 2023, "Development of a hybrid word recognition
system and dataset for the Azerbaijani Sign Language dactyl alphabet"
(Speech Communication, vol. 153). That paper is paywalled and its exact
architecture (classifier types, precise confusion clusters, feature layout)
could not be verified against the source text — only two facts were
confirmed from the companion open-access AzSLD dataset paper: the alphabet
has 24 static + 8 dynamic letters, and the dataset totals ~13-30k samples.
Two corrections were made to the cluster spec handed down for this build:
  - the letter 'K' was missing from every cluster -> added to Cluster 6.
  - the requested "8 dynamic letters" list (Ö,Ü,Ç,İ,Z,D,K) only has 7 items
    and doesn't match the dataset's own structure; the actual 8 letters
    with ~800 sequential-video-frame samples (vs ~40 for the rest) are
    Ç, D, G, K, Ö, Ü, Y, Z — used here as the empirically-grounded dynamic set.
This is an original two-level MLP dispatcher/sub-classifier design built to
the spirit of the brief, not a verified line-for-line paper reproduction.

Pipeline:
  1. Walk every letter folder. Detect which files are genuine sequential
     video frames (e.g. "D_40234.jpg") vs independently crowdsourced photos
     (e.g. "ID--745878352--__15c077c0....jpg" from the JestDiliBot Telegram
     collector) — only sequential frames can yield a real motion signal.
  2. Per image: MediaPipe hand landmarks -> wrist-origin/scale-normalized,
     handedness-mirrored to a canonical "Right hand" -> 63 coords + 15 finger
     joint angles + 4 fingertip-gap distances + 2 velocity components
     (computed from the true frame-number gap within a burst; 0 outside one).
  3. Train a Level-1 MLP dispatcher (80-dim -> 6 confusion clusters) and one
     Level-2 MLP per cluster (2/3-way pairs use a focused 19-dim angle+gap
     feature set; the 21-letter "everything else" cluster uses the full
     80-dim vector). 5-fold stratified CV reported for both levels AND for
     the full dispatch-then-classify pipeline end to end.
  4. Export scalers + MLP weights + cluster map + CV report to
     public/models/azsl_hierarchical_model.json for the browser to run
     natively (must live under public/ so `vite build` copies it into dist/).

Usage:
    python scripts/extract_azsl_model.py
Requires: mediapipe, opencv-python, numpy, scikit-learn (see requirements.txt)
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import cv2
import numpy as np

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as exc:
    sys.exit(f"mediapipe is required: {exc}\npip install -r scripts/requirements.txt")

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    sys.exit(f"scikit-learn is required: {exc}\npip install -r scripts/requirements.txt")

# ============================================================================
# Alphabet, clusters, landmark layout
# ============================================================================
AZ_ALPHABET = [
    "A", "B", "C", "Ç", "D", "E", "Ə", "F", "G", "Ğ", "H", "X", "I", "İ",
    "J", "K", "Q", "L", "M", "N", "O", "Ö", "P", "R", "S", "Ş", "T", "U",
    "Ü", "V", "Y", "Z",
]

# The 8 letters with genuine sequential-video-frame coverage in the dataset
# (~800 samples each vs ~40 for the rest) — used as the "dynamic" set instead
# of the brief's inconsistent 7-item list (see provenance note above).
DYNAMIC_LETTERS = ["Ç", "D", "G", "K", "Ö", "Ü", "Y", "Z"]

CLUSTERS = {
    1: ["R", "X"],
    2: ["C", "Ç", "J"],
    3: ["O", "Ö"],
    4: ["L", "P"],
    5: ["M", "T"],
    6: ["A", "E", "Ə", "B", "D", "F", "G", "Ğ", "H", "I", "İ", "K", "N", "Q",
        "S", "Ş", "U", "Ü", "V", "Y", "Z"],  # 'K' added — missing from the brief
}
LETTER_TO_CLUSTER = {letter: cid for cid, letters in CLUSTERS.items() for letter in letters}
assert sorted(LETTER_TO_CLUSTER) == sorted(AZ_ALPHABET), "cluster map must cover the full alphabet exactly once"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
# Genuine sequential-frame filenames look like "D_40234.jpg" / "AE108.jpg".
# Crowdsourced submissions look like "ID--745878352--__15c077c0-....jpg" and
# must NOT be mistaken for frame sequences.
SEQUENTIAL_NAME_RE = re.compile(r"^[A-Za-zÀ-ɏ]*_?(\d+)\.(jpg|jpeg|png|bmp|webp)$", re.IGNORECASE)
BURST_GAP_THRESHOLD = 50  # frame-number gap beyond which we assume a new recording session

LM = {
    "WRIST": 0,
    "THUMB_CMC": 1, "THUMB_MCP": 2, "THUMB_IP": 3, "THUMB_TIP": 4,
    "INDEX_MCP": 5, "INDEX_PIP": 6, "INDEX_TIP": 8,
    "MIDDLE_MCP": 9, "MIDDLE_PIP": 10, "MIDDLE_TIP": 12,
    "RING_MCP": 13, "RING_PIP": 14, "RING_TIP": 16,
    "PINKY_MCP": 17, "PINKY_PIP": 18, "PINKY_TIP": 20,
}
NUM_LANDMARKS = 21

# Per-finger (base, j1, j2, tip) landmark index quadruples, in a fixed order
# shared with js/gestures.js so the 15 joint angles line up 1:1.
FINGERS = [
    ("thumb", LM["WRIST"], LM["THUMB_MCP"], LM["THUMB_IP"], LM["THUMB_TIP"]),
    ("index", LM["WRIST"], LM["INDEX_MCP"], LM["INDEX_PIP"], LM["INDEX_TIP"]),
    ("middle", LM["WRIST"], LM["MIDDLE_MCP"], LM["MIDDLE_PIP"], LM["MIDDLE_TIP"]),
    ("ring", LM["WRIST"], LM["RING_MCP"], LM["RING_PIP"], LM["RING_TIP"]),
    ("pinky", LM["WRIST"], LM["PINKY_MCP"], LM["PINKY_PIP"], LM["PINKY_TIP"]),
]
MIDDLE_J1, MIDDLE_J2 = LM["MIDDLE_MCP"], LM["MIDDLE_PIP"]

TIP_PAIRS = [
    (LM["THUMB_TIP"], LM["INDEX_TIP"]),
    (LM["INDEX_TIP"], LM["MIDDLE_TIP"]),
    (LM["MIDDLE_TIP"], LM["RING_TIP"]),
    (LM["RING_TIP"], LM["PINKY_TIP"]),
]

FEATURE_LAYOUT = {
    "coords": {"offset": 0, "length": 63, "description": "21 landmarks x (x,y,z), wrist-origin, scale-normalized, handedness-mirrored to Right"},
    "angles": {"offset": 63, "length": 15, "description": "5 fingers x 3 angles (base-flex, tip-flex, spread-vs-middle), degrees"},
    "tipDistances": {"offset": 78, "length": 4, "description": "thumb-index, index-middle, middle-ring, ring-pinky tip distances"},
    "velocity": {"offset": 82, "length": 2, "description": "(dX, dY) of wrist position per video frame within a detected burst, 0 outside one"},
}
FULL_VECTOR_LENGTH = 84
LEVEL2_NARROW_INDICES = list(range(63, 82))  # angles + tip distances only, for the 2/3-letter clusters


# ============================================================================
# Geometry helpers
# ============================================================================
def vec_sub(a, b):
    return a - b


def angle_between(v1, v2):
    mags = np.linalg.norm(v1) * np.linalg.norm(v2)
    if mags < 1e-9:
        return 0.0
    cos = np.clip(np.dot(v1, v2) / mags, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def normalize_landmarks(landmarks_xyz, mirror_x):
    wrist = landmarks_xyz[LM["WRIST"]]
    shifted = landmarks_xyz - wrist
    scale = np.linalg.norm(shifted[LM["MIDDLE_MCP"]])
    if scale < 1e-6:
        scale = 1e-6
    normalized = shifted / scale
    if mirror_x:
        normalized = normalized.copy()
        normalized[:, 0] *= -1
    return normalized


def joint_angles(coords):
    """15 angles: for each of 5 fingers, (base-flex, tip-flex, spread-vs-middle)."""
    out = np.zeros(15, dtype=np.float64)
    middle_dir = coords[MIDDLE_J2] - coords[MIDDLE_J1]
    for i, (_, base, j1, j2, tip) in enumerate(FINGERS):
        base_flex = angle_between(coords[base] - coords[j1], coords[j2] - coords[j1])
        tip_flex = angle_between(coords[j1] - coords[j2], coords[tip] - coords[j2])
        this_dir = coords[j2] - coords[j1]
        spread = angle_between(this_dir, middle_dir)
        out[i * 3: i * 3 + 3] = [base_flex, tip_flex, spread]
    return out


def tip_distances(coords):
    return np.array([np.linalg.norm(coords[a] - coords[b]) for a, b in TIP_PAIRS], dtype=np.float64)


def build_feature_vector(coords, velocity_xy):
    return np.concatenate([coords.flatten(), joint_angles(coords), tip_distances(coords), velocity_xy])


# ============================================================================
# File discovery + burst grouping
# ============================================================================
def imread_unicode(path):
    with open(path, "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def list_class_files(folder):
    """Returns (sequential, standalone): sequential is [(frame_no, filename), ...]
    sorted by frame number (genuine burst candidates); standalone is filenames
    with no reliable temporal relationship to any other file (crowdsourced)."""
    sequential, standalone = [], []
    for fname in os.listdir(folder):
        if not fname.lower().endswith(IMAGE_EXTENSIONS):
            continue
        m = SEQUENTIAL_NAME_RE.match(fname)
        if m:
            sequential.append((int(m.group(1)), fname))
        else:
            standalone.append(fname)
    sequential.sort(key=lambda t: t[0])
    return sequential, standalone


# ============================================================================
# MediaPipe extraction
# ============================================================================
def build_landmarker(model_path, min_confidence):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=1,
        min_hand_detection_confidence=min_confidence,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_hand(landmarker, path):
    img_bgr = imread_unicode(path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None
    raw_xyz = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float64)
    mirror_x = bool(result.handedness) and result.handedness[0][0].category_name == "Left"
    return raw_xyz, mirror_x


def extract_letter(landmarker, letter, folder):
    """Runs detection on every file, computing genuine within-burst velocity
    for sequential frames and zero velocity for standalone crowdsourced photos."""
    sequential, standalone = list_class_files(folder)
    records = []

    prev_frame_no, prev_wrist = None, None
    for frame_no, fname in sequential:
        detection = detect_hand(landmarker, os.path.join(folder, fname))
        if detection is None:
            prev_frame_no, prev_wrist = None, None  # break the chain on a miss
            continue
        raw_xyz, mirror_x = detection
        wrist_xy = raw_xyz[LM["WRIST"], :2]

        velocity = np.zeros(2, dtype=np.float64)
        if prev_frame_no is not None:
            gap = frame_no - prev_frame_no
            if 0 < gap <= BURST_GAP_THRESHOLD:
                velocity = (wrist_xy - prev_wrist) / gap
        prev_frame_no, prev_wrist = frame_no, wrist_xy

        coords = normalize_landmarks(raw_xyz, mirror_x)
        # Velocity is mirrored consistently with the coordinate frame.
        v = velocity.copy()
        if mirror_x:
            v[0] *= -1
        records.append(build_feature_vector(coords, v))

    for fname in standalone:
        detection = detect_hand(landmarker, os.path.join(folder, fname))
        if detection is None:
            continue
        raw_xyz, mirror_x = detection
        coords = normalize_landmarks(raw_xyz, mirror_x)
        records.append(build_feature_vector(coords, np.zeros(2, dtype=np.float64)))

    return records, len(sequential) + len(standalone)


def extract_all(data_dir, model_path, min_confidence, max_per_class, seed):
    landmarker = build_landmarker(model_path, min_confidence)
    available = {d: os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))}
    rng = np.random.RandomState(seed)

    X, y_letter = [], []
    summary = []
    for letter in AZ_ALPHABET:
        folder = available.get(letter)
        if not folder:
            print(f"  [WARN] no folder for '{letter}'")
            summary.append((letter, 0, 0))
            continue
        t0 = time.time()
        records, attempted = extract_letter(landmarker, letter, folder)
        if max_per_class and len(records) > max_per_class:
            idx = rng.choice(len(records), size=max_per_class, replace=False)
            records = [records[i] for i in idx]
        for r in records:
            X.append(r)
            y_letter.append(letter)
        summary.append((letter, attempted, len(records)))
        print(f"  [OK] {letter}: {len(records)}/{attempted} usable ({time.time() - t0:.1f}s)")

    return np.array(X, dtype=np.float64), np.array(y_letter), summary


# ============================================================================
# MLP export helpers
# ============================================================================
def export_scaler(scaler):
    return {"mean": scaler.mean_.round(8).tolist(), "std": scaler.scale_.round(8).tolist()}


def export_mlp(clf):
    """sklearn collapses binary classification to a single logistic output
    neuron — normalize that away so JS always sees an explicit per-class
    output layer with a documented activation."""
    n_classes = len(clf.classes_)
    layers = [{"weights": w.round(8).tolist(), "biases": b.round(8).tolist()}
              for w, b in zip(clf.coefs_, clf.intercepts_)]
    output_activation = "sigmoid_binary" if n_classes == 2 and len(layers[-1]["biases"]) == 1 else "softmax"
    return {
        "classes": clf.classes_.tolist(),  # .tolist() (not list()) unwraps numpy.int64 into plain JSON-serializable ints
        "layers": layers,
        "hiddenActivation": "relu",
        "outputActivation": output_activation,
    }


# ============================================================================
# Training + cross-validation
# ============================================================================
def make_mlp(hidden, alpha, seed):
    # early_stopping=True crashes here: sklearn's internal held-out validation
    # score calls np.isnan() on predicted labels, which breaks for our string
    # class labels (Azerbaijani letters) — works fine for Level-1's numeric
    # cluster ids, which is why only Level-2 fitting crashed. Training-loss
    # based convergence (tol/n_iter_no_change, still active without
    # early_stopping) is sufficient at this dataset size regardless.
    return MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, max_iter=3000,
                          early_stopping=False, n_iter_no_change=25, random_state=seed)


def cross_validate(X, y, hidden, alpha, seed, folds=5):
    counts = {c: int(np.sum(y == c)) for c in np.unique(y)}
    min_count = min(counts.values())
    k = min(folds, min_count)
    if k < 2:
        return None  # not enough samples in the smallest class to fold at all
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        clf = make_mlp(hidden, alpha, seed)
        clf.fit(scaler.transform(X[train_idx]), y[train_idx])
        accs.append(float(clf.score(scaler.transform(X[test_idx]), y[test_idx])))
    return {"folds": k, "perFold": [round(a, 4) for a in accs], "mean": round(float(np.mean(accs)), 4)}


def end_to_end_cross_validate(X, y_letter, y_cluster, seed, folds=5):
    """The metric that matters: dispatch through Level-1, then classify with
    the matching Level-2 model, fresh-trained per fold — compare the FINAL
    predicted letter against ground truth (not just per-level accuracy)."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs = []
    for train_idx, test_idx in skf.split(X, y_letter):
        scaler80 = StandardScaler().fit(X[train_idx])
        level1 = make_mlp((48, 24), 1e-3, seed)
        level1.fit(scaler80.transform(X[train_idx]), y_cluster[train_idx])

        level2_models = {}
        for cid, letters in CLUSTERS.items():
            mask = np.isin(y_letter[train_idx], letters)
            if cid == 6:
                sub_scaler = StandardScaler().fit(X[train_idx][mask])
                clf = make_mlp((64, 32), 1e-3, seed)
                clf.fit(sub_scaler.transform(X[train_idx][mask]), y_letter[train_idx][mask])
            else:
                sub_X = X[train_idx][:, LEVEL2_NARROW_INDICES][mask]
                sub_scaler = StandardScaler().fit(sub_X)
                clf = make_mlp((16,), 1e-2, seed)
                clf.fit(sub_scaler.transform(sub_X), y_letter[train_idx][mask])
            level2_models[cid] = (clf, sub_scaler)

        pred_clusters = level1.predict(scaler80.transform(X[test_idx]))
        final_preds = []
        for i, cid in enumerate(pred_clusters):
            clf, sub_scaler = level2_models[cid]
            if cid == 6:
                feat = scaler80.transform(X[test_idx][i:i + 1])
            else:
                feat = sub_scaler.transform(X[test_idx][i:i + 1][:, LEVEL2_NARROW_INDICES])
            final_preds.append(clf.predict(feat)[0])

        accs.append(float(np.mean(np.array(final_preds) == y_letter[test_idx])))
    return {"folds": folds, "perFold": [round(a, 4) for a in accs], "mean": round(float(np.mean(accs)), 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/AzSLD_Fingerspelling")
    parser.add_argument("--model-path", default="public/models/hand_landmarker.task")
    parser.add_argument("--output", default="public/models/azsl_hierarchical_model.json")
    parser.add_argument("--max-per-class", type=int, default=250)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        sys.exit(f"Dataset directory not found: {args.data_dir}")
    if not os.path.isfile(args.model_path):
        sys.exit(f"Model file not found: {args.model_path}")

    print(f"Dataset:       {args.data_dir}")
    print(f"Max/class:     {args.max_per_class}")
    print(f"Dynamic set:   {', '.join(DYNAMIC_LETTERS)} (empirically derived, see docstring)\n")

    print("=== Step 1: Feature extraction ===")
    X, y_letter, summary = extract_all(args.data_dir, args.model_path, args.min_confidence, args.max_per_class, args.seed)
    y_cluster = np.array([LETTER_TO_CLUSTER[l] for l in y_letter])
    print(f"\nTotal usable samples: {len(y_letter)} across {len(set(y_letter))} letters\n")

    print("=== Step 2: Level-1 dispatcher (6-way cluster classifier) ===")
    scaler80 = StandardScaler().fit(X)
    X_scaled = scaler80.transform(X)
    level1_cv = cross_validate(X, y_cluster, (48, 24), 1e-3, args.seed)
    print(f"  5-fold CV accuracy: {level1_cv['mean']:.4f}  (per-fold {level1_cv['perFold']})")
    level1_final = make_mlp((48, 24), 1e-3, args.seed)
    level1_final.fit(X_scaled, y_cluster)

    print("\n=== Step 3: Level-2 sub-classifiers ===")
    level2_final = {}
    level2_cv_report = {}
    scaler_by_cluster = {}
    for cid, letters in CLUSTERS.items():
        mask = y_cluster == cid
        label = "+".join(letters) if cid != 6 else f"main set ({len(letters)} letters)"
        if cid == 6:
            cv = cross_validate(X[mask], y_letter[mask], (64, 32), 1e-3, args.seed)
            scaler = StandardScaler().fit(X[mask])
            clf = make_mlp((64, 32), 1e-3, args.seed)
            clf.fit(scaler.transform(X[mask]), y_letter[mask])
        else:
            sub_X = X[mask][:, LEVEL2_NARROW_INDICES]
            cv = cross_validate(sub_X, y_letter[mask], (16,), 1e-2, args.seed)
            scaler = StandardScaler().fit(sub_X)
            clf = make_mlp((16,), 1e-2, args.seed)
            clf.fit(scaler.transform(sub_X), y_letter[mask])
        scaler_by_cluster[cid] = scaler
        level2_final[cid] = clf
        level2_cv_report[cid] = cv
        acc_str = f"{cv['mean']:.4f}" if cv else "N/A (too few samples to fold)"
        print(f"  Cluster {cid} [{label}]: 5-fold CV accuracy = {acc_str}")

    print("\n=== Step 4: End-to-end pipeline cross-validation (dispatch -> classify) ===")
    e2e_cv = end_to_end_cross_validate(X, y_letter, y_cluster, args.seed)
    print(f"  5-fold end-to-end accuracy: {e2e_cv['mean']:.4f}  (per-fold {e2e_cv['perFold']})")

    print("\n=== Exporting model ===")
    clusters_export = {}
    for cid, letters in CLUSTERS.items():
        entry = {"letters": letters, "model": export_mlp(level2_final[cid]), "scaler": export_scaler(scaler_by_cluster[cid])}
        entry["featureIndices"] = LEVEL2_NARROW_INDICES if cid != 6 else list(range(FULL_VECTOR_LENGTH))
        clusters_export[str(cid)] = entry

    output = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "alphabet": AZ_ALPHABET,
        "dynamicLetters": DYNAMIC_LETTERS,
        "featureVectorLength": FULL_VECTOR_LENGTH,
        "featureLayout": FEATURE_LAYOUT,
        "provenanceNote": (
            "Two-level MLP dispatcher inspired by Hasanov et al. 2023's clustered-model "
            "concept; exact paper architecture unverified (paywalled). 'K' added to "
            "Cluster 6 (missing from spec); dynamic-letter set corrected to match the "
            "dataset's own 8 high-volume sequential-frame folders."
        ),
        "level1": {"scaler": export_scaler(scaler80), "model": export_mlp(level1_final), "crossValidation": level1_cv},
        "clusters": clusters_export,
        "crossValidation": {"level2ByCluster": level2_cv_report, "endToEnd": e2e_cv},
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}")

    print("\nPer-class sample summary:")
    print(f"{'Letter':<8}{'Attempted':<12}{'Usable':<8}")
    for letter, attempted, usable in summary:
        print(f"{letter:<8}{attempted:<12}{usable:<8}")


if __name__ == "__main__":
    main()

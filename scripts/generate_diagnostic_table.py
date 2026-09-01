#!/usr/bin/env python3
"""
Compute the exact diagnostic table requested by user across all 7 dynamic classes.
"""

import sys, os
sys.path.insert(0, ".")
import numpy as np

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from scripts.dynamic_dataset import DYNAMIC_CLASSES
from scripts.dynamic_model import DynamicGestureRecognizer
from scripts.static_model import StaticHierarchicalModel

cache = np.load("data/dynamic_landmarks_cache.npz", allow_pickle=True)
X = cache["X"]
y = cache["y"]

recognizer = DynamicGestureRecognizer("models/dynamic_model.pt", device="cpu")
static_model = StaticHierarchicalModel("public/models/azsl_hierarchical_model.json")

STATIC_ONLY = set([
    'A', 'B', 'E', 'Ə', 'F', 'G', 'Ğ', 'H', 'X', 'I', 'İ',
    'J', 'K', 'Q', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V'
])

print("==========================================================================================")
print("REQUIRED DIAGNOSTIC TABLE EVALUATION ACROSS ALL 7 CLASSES")
print("==========================================================================================")

table_rows = []

for c in DYNAMIC_CLASSES:
    c_idx = DYNAMIC_CLASSES.index(c)
    indices = np.where(y == c_idx)[0]
    
    # Analyze the streaming trajectory for the first representative sample
    seq = X[indices[0]] # (20, 63)
    stream = np.concatenate([
        np.repeat(seq[0:1], 15, axis=0),
        seq,
        np.repeat(seq[-1:], 15, axis=0)
    ], axis=0) # (50, 63)
    
    buf = []
    recent_e = []
    
    # Metrics to capture
    triggered = False
    complete_20 = False
    onnx_pred = None
    p_z = 0.0
    p_y = 0.0
    dyn_conf = 0.0
    stat_pred = None
    final_pred = None
    root_cause = None
    
    for t in range(len(stream)):
        frame = stream[t]
        e = float(np.mean(np.abs(frame - buf[-1]))) if len(buf) > 0 else 0.0
        buf.append(frame)
        if len(buf) > 20: buf.pop(0)
        recent_e.append(e)
        if len(recent_e) > 10: recent_e.pop(0)
        
        avg_e = np.mean(recent_e)
        cum_disp = float(np.sum(np.linalg.norm(np.diff(buf, axis=0), axis=1))) if len(buf) >= 2 else 0.0
        
        s_res = static_model.predict_from_landmarks(frame.reshape(21, 3), mirror_x=False, velocity_xy=np.zeros(2))
        
        # Current index.html thresholds:
        is_moving = avg_e >= 0.020
        has_traj = len(buf) >= 20 and cum_disp >= 4.8 and is_moving
        
        if len(buf) == 20:
            complete_20 = True
            d_res = recognizer.predict_sequence(np.array(buf))
            
            # Mid gesture peak evaluation (t=25..30)
            if 25 <= t <= 30 and onnx_pred is None:
                triggered = has_traj
                onnx_pred = d_res["label"]
                dyn_conf = d_res["confidence"]
                p_z = d_res["probabilities"].get("Z", 0.0)
                p_y = d_res["probabilities"].get("Y", 0.0)
                stat_pred = s_res["label"]
                
                # Arbitration logic from index.html:
                is_static_protected = (s_res['confidence'] >= 0.70 and s_res['label'] in STATIC_ONLY and cum_disp < 10.0)
                if has_traj and d_res['confidence'] >= 0.65 and not is_static_protected:
                    final_pred = d_res['label']
                else:
                    final_pred = s_res['label']
                    
    table_rows.append({
        "letter": c,
        "triggered": "Yes" if triggered else "No (CumDisp<4.8)",
        "complete_20": "Yes" if complete_20 else "No",
        "onnx_pred": onnx_pred,
        "p_z": p_z,
        "p_y": p_y,
        "dyn_conf": dyn_conf,
        "stat_pred": stat_pred,
        "final_pred": final_pred,
    })

print("%-6s | %-16s | %-12s | %-10s | %-8s | %-8s | %-8s | %-10s | %-10s" % (
    "Letter", "Dynamic Trig?", "20-Frame?", "ONNX Pred", "P(Z)", "P(Y)", "Dyn Conf", "Stat Pred", "Final Pred"
))
print("-" * 102)
for r in table_rows:
    print("%-6s | %-16s | %-12s | %-10s | %8.3f | %8.3f | %8.2f | %-10s | %-10s" % (
        r["letter"], r["triggered"], r["complete_20"], str(r["onnx_pred"]),
        r["p_z"], r["p_y"], r["dyn_conf"], str(r["stat_pred"]), str(r["final_pred"])
    ))


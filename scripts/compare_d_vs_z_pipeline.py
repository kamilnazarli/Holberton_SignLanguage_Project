#!/usr/bin/env python3
"""
Step 5 & 7: Controlled comparative test between D (control case) and Z.
Tests:
1. ONNX vs PyTorch inference on all Z windows in cache.
2. Step-by-step pipeline trace for D vs Z:
   - landmark magnitude
   - motion energy
   - trajectory length
   - buffer timing
   - number of inference windows
   - model confidence
   - arbitration
   - commit timing
"""

import sys, os
sys.path.insert(0, ".")
import numpy as np
from scripts.dynamic_dataset import DYNAMIC_CLASSES
from scripts.dynamic_model import DynamicGestureRecognizer
from scripts.static_model import StaticHierarchicalModel

cache = np.load("data/dynamic_landmarks_cache.npz", allow_pickle=True)
X = cache["X"]
y = cache["y"]

pytorch_model = DynamicGestureRecognizer("models/dynamic_model.pt", device="cpu")
static_model = StaticHierarchicalModel("public/models/azsl_hierarchical_model.json")

print("==========================================================================================")
print("1. PYTORCH PREDICTIONS ON ALL 102 Z SEQUENCES")
print("==========================================================================================")

z_indices = np.where(y == DYNAMIC_CLASSES.index("Z"))[0]
d_indices = np.where(y == DYNAMIC_CLASSES.index("D"))[0]

z_preds = []
z_confs = []

for idx in z_indices:
    seq = X[idx] # (20, 63)
    py_res = pytorch_model.predict_sequence(seq)
    z_preds.append(py_res["label"])
    z_confs.append(py_res["confidence"])

print(f"Accuracy on Z: {np.mean([1 if p == 'Z' else 0 for p in z_preds])*100:.1f}%")
print(f"Average Confidence on Z: {np.mean(z_confs):.4f}")

print("\n==========================================================================================")
print("2. D (CONTROL) VS Z PIPELINE STAGE TRACE")
print("==========================================================================================")

def simulate_pipeline_trace(class_name, seq_idx):
    c_idx = DYNAMIC_CLASSES.index(class_name)
    indices = np.where(y == c_idx)[0]
    seq = X[indices[seq_idx]] # (20, 63)
    
    # Simulate a stream: 10 neutral frames + 20 gesture frames + 10 landing frames (40 frames total)
    stream = np.concatenate([
        np.repeat(seq[0:1], 10, axis=0),
        seq,
        np.repeat(seq[-1:], 10, axis=0)
    ], axis=0) # (40, 63)
    
    # Trace per frame:
    # - Landmark magnitude (index tip)
    # - Frame delta energy
    # - Cumulative displacement in buffer
    # - Dynamic prediction & conf
    # - Static prediction & conf
    # - Arbitration mode & final label
    buffer = []
    recent_energies = []
    hangover = 0
    hold_count = 0
    last_label = None
    committed = None
    commit_time = None
    
    trace_rows = []
    
    for t in range(len(stream)):
        frame = stream[t]
        
        # Energy relative to prev frame
        if len(buffer) > 0:
            e = float(np.mean(np.abs(frame - buffer[-1])))
        else:
            e = 0.0
            
        buffer.append(frame)
        if len(buffer) > 20:
            buffer.pop(0)
            
        recent_energies.append(e)
        if len(recent_energies) > 10:
            recent_energies.pop(0)
            
        avg_e = np.mean(recent_energies) if recent_energies else 0.0
        cum_disp = float(np.sum(np.linalg.norm(np.diff(buffer, axis=0), axis=1))) if len(buffer) >= 2 else 0.0
        
        # Static prediction
        s_res = static_model.predict_from_landmarks(frame.reshape(21, 3), mirror_x=False, velocity_xy=np.zeros(2))
        
        # Dynamic prediction
        dyn_pred = None
        dyn_conf = 0.0
        is_moving = avg_e >= 0.020
        buffer_filled = len(buffer) >= 20
        has_traj = buffer_filled and cum_disp >= 4.8 and is_moving
        
        in_hangover = False
        if has_traj:
            hangover = 10
        else:
            if hangover > 0:
                hangover -= 1
                in_hangover = True
                
        if buffer_filled and (has_traj or in_hangover):
            dyn_res = pytorch_model.predict_sequence(np.array(buffer))
            dyn_pred = dyn_res["label"]
            dyn_conf = dyn_res["confidence"]
            
        # Arbitration
        is_static_protected = (s_res['confidence'] >= 0.70 and s_res['label'] in set(['A', 'B', 'M', 'K', 'T', 'S']) and cum_disp < 10.0)
        
        if (has_traj or in_hangover) and dyn_pred and dyn_conf >= 0.65 and not is_static_protected:
            mode = "DYNAMIC"
            final_label = dyn_pred
            final_conf = dyn_conf
        else:
            mode = "STATIC"
            final_label = s_res['label']
            final_conf = s_res['confidence']
            
        # Commit logic
        if final_label == last_label:
            hold_count += 1
        else:
            last_label = final_label
            hold_count = 1
            
        req_frames = 11 if mode == "DYNAMIC" else 36
        if hold_count >= req_frames and committed is None:
            committed = final_label
            commit_time = t * 33 # ms
            
        trace_rows.append({
            "t": t,
            "time_ms": t * 33,
            "avg_e": avg_e,
            "cum_disp": cum_disp,
            "dyn_pred": dyn_pred,
            "dyn_conf": dyn_conf,
            "stat_pred": s_res['label'],
            "stat_conf": s_res['confidence'],
            "mode": mode,
            "final": final_label,
            "committed": committed,
        })
        
    return trace_rows, committed, commit_time

trace_d, comm_d, time_d = simulate_pipeline_trace("D", 0)
trace_z, comm_z, time_z = simulate_pipeline_trace("Z", 0)

print(f"CONTROL 'D': Committed='{comm_d}' at t={time_d} ms")
print(f"TARGET  'Z': Committed='{comm_z}' at t={time_z} ms")

print("\n--- SAMPLE FRAME-BY-FRAME COMPARISON (t=15 to 30) ---")
print("%-4s | %-12s | %-12s | %-12s | %-12s | %-10s | %-10s" % ("t", "D Energy/Cum", "Z Energy/Cum", "D Dyn (Conf)", "Z Dyn (Conf)", "D Final", "Z Final"))
print("-" * 82)
for t in range(15, 30):
    rd = trace_d[t]
    rz = trace_z[t]
    d_e_str = f"{rd['avg_e']:.3f}/{rd['cum_disp']:.1f}"
    z_e_str = f"{rz['avg_e']:.3f}/{rz['cum_disp']:.1f}"
    d_dyn = f"{rd['dyn_pred'] or '-'}:{rd['dyn_conf']:.2f}"
    z_dyn = f"{rz['dyn_pred'] or '-'}:{rz['dyn_conf']:.2f}"
    print("%-4d | %-12s | %-12s | %-12s | %-12s | %-10s | %-10s" % (
        t, d_e_str, z_e_str, d_dyn, z_dyn, f"{rd['mode']} {rd['final']}", f"{rz['mode']} {rz['final']}"
    ))

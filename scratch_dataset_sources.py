import sys, os, glob
import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from scripts.dynamic_dataset import LandmarkerWrapper
from scripts.static_model import (
    normalize_landmarks,
    build_feature_vector_84,
    apply_scaler,
    mlp_forward,
)

landmarker = LandmarkerWrapper("public/models/hand_landmarker.task", min_confidence=0.5)

# Let's inspect the actual G dataset images vs Ş dataset images:
# How many signers are in G?
# In G, all images are named G_35604.jpg, G_35620.jpg ... from ONE SINGLE burst/video!
g_files = sorted(glob.glob("data/AzSLD_Fingerspelling/G/*.jpg"))
print(f"Number of G files: {len(g_files)}")
print(f"File pattern for G: {g_files[0]} ... {g_files[-1]}")

# How many signers in Ş?
sh_files = sorted(glob.glob("data/AzSLD_Fingerspelling/Ş/*.jpg"))
print(f"Number of Ş files: {len(sh_files)}")
print(f"File sample for Ş: {[os.path.basename(f) for f in sh_files[:5]]}")

# How many signers in H?
h_files = sorted(glob.glob("data/AzSLD_Fingerspelling/H/*.jpg"))
print(f"Number of H files: {len(h_files)}")
print(f"File sample for H: {[os.path.basename(f) for f in h_files[:5]]}")

# How many signers in P?
p_files = sorted(glob.glob("data/AzSLD_Fingerspelling/P/*.jpg"))
print(f"Number of P files: {len(p_files)}")
print(f"File sample for P: {[os.path.basename(f) for f in p_files[:5]]}")

# How many signers in J?
j_files = sorted(glob.glob("data/AzSLD_Fingerspelling/J/*.jpg"))
print(f"Number of J files: {len(j_files)}")
print(f"File sample for J: {[os.path.basename(f) for f in j_files[:5]]}")

# How many signers in Ç?
ch_files = sorted(glob.glob("data/AzSLD_Fingerspelling/Ç/*.jpg"))
print(f"Number of Ç files: {len(ch_files)}")
print(f"File sample for Ç: {[os.path.basename(f) for f in ch_files[:5]]}")

# How many signers in F?
f_files = sorted(glob.glob("data/AzSLD_Fingerspelling/F/*.jpg"))
print(f"Number of F files: {len(f_files)}")
print(f"File sample for F: {[os.path.basename(f) for f in f_files[:5]]}")


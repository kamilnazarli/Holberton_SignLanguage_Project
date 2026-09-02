import os, glob
import cv2

for l in ["G", "Ş", "H", "P", "J", "Ç", "F"]:
    folder = os.path.join("data", "AzSLD_Fingerspelling", l)
    files = sorted(glob.glob(os.path.join(folder, "*.*")))[:5]
    print(f"=== Class {l} ===")
    for f in files:
        img = cv2.imread(f)
        if img is not None:
            h, w, c = img.shape
            print(f"  {os.path.basename(f)}: size={w}x{h}")
        else:
            print(f"  {os.path.basename(f)}: could not load directly")


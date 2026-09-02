import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

data_dir = "data/AzSLD_Fingerspelling"
letters = sorted(os.listdir(data_dir))

print(f"{'Letter':<8} | {'Total Files':<12} | {'Image Files':<12} | {'Sample Files'}")
print("-" * 60)

for letter in letters:
    lpath = os.path.join(data_dir, letter)
    if os.path.isdir(lpath):
        files = os.listdir(lpath)
        img_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        sample = img_files[:3]
        if letter in ["G", "H", "J", "Ş", "P", "F", "Ç", "K", "B", "A", "E"]:
            print(f"{letter:<8} | {len(files):<12} | {len(img_files):<12} | {sample}")

print("\n--- Summary for all 32 classes ---")
counts = {}
for letter in letters:
    lpath = os.path.join(data_dir, letter)
    if os.path.isdir(lpath):
        files = os.listdir(lpath)
        img_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        counts[letter] = len(img_files)

print("Classes with < 50 images:", {k: v for k, v in counts.items() if v < 50})
print("Classes with >= 500 images:", {k: v for k, v in counts.items() if v >= 500})


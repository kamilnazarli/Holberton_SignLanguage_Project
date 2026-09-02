import os, glob
import sys

sys.stdout.reconfigure(encoding="utf-8")

data_dir = "data/AzSLD_Fingerspelling"
letters = sorted(os.listdir(data_dir))

print(f"{'Letter':<6} | {'Count':<6} | {'Prefix/Pattern':<35} | {'Source Type'}")
print("-" * 75)

for l in letters:
    lpath = os.path.join(data_dir, l)
    if not os.path.isdir(lpath): continue
    files = [f for f in os.listdir(lpath) if f.lower().endswith(('.jpg', '.png'))]
    if not files: continue
    
    # Check prefixes
    has_id = any(f.startswith("ID--") for f in files)
    has_burst = any("_" in f and f.split("_")[0] == l for f in files)
    has_letter_num = any(f[0].upper() == l and f[1:].split(".")[0].isdigit() for f in files)
    
    types = []
    if has_id: types.append("JestDiliBot (Telegram)")
    if has_burst: types.append("Video Burst")
    if has_letter_num: types.append("ASL/Synthetic LetterNum")
    
    sample_prefix = files[0][:30]
    print(f"{l:<6} | {len(files):<6} | {sample_prefix:<35} | {', '.join(types)}")


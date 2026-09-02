import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("public/models/azsl_hierarchical_model.json", "r", encoding="utf-8") as f:
    model = json.load(f)

print("Level 1 Classes (Clusters):", model["level1"]["model"]["classes"])
print("\nLevel 2 Clusters:")
for cid, cinfo in model["clusters"].items():
    print(f"Cluster {cid}: letters = {cinfo['letters']}")
    print(f"   Model classes = {cinfo['model']['classes']}")
    print(f"   Feature count = {len(cinfo['featureIndices'])}")

all_letters = []
for cid, cinfo in model["clusters"].items():
    all_letters.extend(cinfo["model"]["classes"])

print(f"\nTotal letters in model: {len(all_letters)}")
print("Are G, H, J present?")
for target in ["G", "H", "J", "Ş", "P", "F", "Ç"]:
    found_in = [cid for cid, cinfo in model["clusters"].items() if target in cinfo["model"]["classes"]]
    print(f"  {target}: found in Cluster {found_in}")

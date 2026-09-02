import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

with open("public/models/azsl_hierarchical_model.json", "r", encoding="utf-8") as f:
    m = json.load(f)

print("Generated at:", m.get("generatedAt"))
print("Provenance:", m.get("provenanceNote"))
print("Level 1 CV:", m.get("level1", {}).get("crossValidation"))
print("Level 2 CV:", m.get("crossValidation", {}).get("level2ByCluster"))
print("End-to-end CV:", m.get("crossValidation", {}).get("endToEnd"))


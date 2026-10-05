import json
import re
from pathlib import Path
from collections import Counter

# 合併規則，有需要再加
REMAP = {
    "fine":                                "healthy",
    "hypersensitivity_allergic_dermatosis": "hypersensitivity_dermatitis",
}

DATA_DIR = Path("merged_data")
SPLITS   = ["train", "val", "test"]

all_names = set()
patched   = {}

for split in SPLITS:
    path = DATA_DIR / split / f"{split}_annotations.json"
    if not path.exists():
        continue

    with open(path, "r", encoding="utf-8") as f:
        anns = json.load(f)

    changed = 0
    for ann in anns:
        old = ann["class_name"]
        new = REMAP.get(old, old)
        if new != old:
            ann["class_name"] = new
            changed += 1

    all_names.update(ann["class_name"] for ann in anns)
    patched[split] = anns
    print(f"{split}: {changed} renamed")

class_names = sorted(all_names)
cls2idx     = {n: i for i, n in enumerate(class_names)}

print(f"\nFinal classes ({len(class_names)}):")
for i, name in enumerate(class_names):
    total = sum(
        sum(1 for a in anns if a["class_name"] == name)
        for anns in patched.values()
    )
    print(f"  {i:2d}  {name:40s}  {total} samples")

for split, anns in patched.items():
    path = DATA_DIR / split / f"{split}_annotations.json"
    for ann in anns:
        ann["label"] = cls2idx[ann["class_name"]]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(anns, f, indent=2, ensure_ascii=False)

print("\n✓ JSON updated")

cfg_path = Path("config_classification.py")
if cfg_path.exists():
    content = cfg_path.read_text(encoding="utf-8")
    content = re.sub(r"NUM_CLASSES\s*=\s*\d+",
                     f"NUM_CLASSES = {len(class_names)}", content)
    content = re.sub(r"CLASS_NAMES\s*=\s*\[.*?\]",
                     f"CLASS_NAMES = {class_names}", content, flags=re.DOTALL)
    content = re.sub(r"CLASS_TO_IDX\s*=\s*\{.*?\}",
                     f"CLASS_TO_IDX = {cls2idx}", content, flags=re.DOTALL)
    cfg_path.write_text(content, encoding="utf-8")
    print("✓ config_classification.py updated")

print("\nDone! Now run: python3 train_classification.py")

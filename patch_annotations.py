"""檢查 dataset_cc_by/ 與 config_classification.py 是否一致（類別順序、檔案存在、標籤範圍）。

舊版用來改名並重寫 config；現在 build_dataset.py 已直接產生統一類別，這支只做檢查，不改任何檔案。
用法：python patch_annotations.py
"""
import json
import sys
from collections import Counter

from config_classification import config

SPLITS = ["train", "val", "test"]


def main():
    ok = True
    cls_file = config.DATA_DIR / "classes.json"
    if not cls_file.exists():
        sys.exit(f"找不到 {cls_file}，請先執行 python build_dataset.py")
    classes = json.load(open(cls_file, encoding="utf-8"))
    if classes != config.CLASS_NAMES:
        print("✗ classes.json 與 config.CLASS_NAMES 不一致：")
        print("  dataset:", classes)
        print("  config :", config.CLASS_NAMES)
        ok = False
    for s in SPLITS:
        d = config.DATA_DIR / s
        f = d / f"{s}_annotations.json"
        if not f.exists():
            print(f"✗ 缺少 {f}")
            ok = False
            continue
        anns = json.load(open(f, encoding="utf-8"))
        missing = [a["image_path"] for a in anns if not (d / a["image_path"]).exists()]
        bad = [a for a in anns if not 0 <= a["label"] < config.NUM_CLASSES
               or config.CLASS_NAMES[a["label"]] != a["class_name"]]
        cnt = Counter(a["class_name"] for a in anns)
        print(f"{s:5s} {len(anns):5d} 張  缺檔 {len(missing)}  標籤不符 {len(bad)}  "
              + " ".join(f"{k[:6]}={cnt[k]}" for k in config.CLASS_NAMES))
        ok &= not missing and not bad
    print("\n✓ 資料集與設定一致" if ok else "\n✗ 有問題，請修正後再訓練")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

"""
重新分層切分 train/val/test，修正「黑邊旋轉殘留圖片集中在 test/val」的分布不一致問題。

背景：audit_image_quality.py 發現 dermatophytosis 的黑邊污染比例在 train(8.0%)、
val(16.9%)、test(31.9%) 之間差了 4 倍，其他類別也有類似但較輕微的偏斜。這代表目前
的 train/val/test 是「先天分布不均」的切分，不是隨機切的，會讓驗證/測試結果失真、
也可能是 dermatophytosis recall 崩潰的成因之一。

做法：
  1. 對全部 7064 張圖重新計算 tiny_crop / rotated_padding 品質旗標。
  2. 每個類別「維持原本 train/val/test 的樣本數」不變（不改變類別分布、不影響
     class weight 計算），只在同一類別內把「乾淨圖」和「黑邊/過小圖」分開，各自
     用固定 seed 打亂後按比例分回三個 split，讓每個 split 裡的黑邊比例趨近全類別
     平均值，而不是像現在一樣集中在 test。
  3. 只搬動「需要換 split」的圖片檔案（rename 到新的 split 資料夾），不修改、
     不刪除任何圖片內容本身。
  4. 執行前完整備份 merged_data 到 merged_data_backup_pre_resplit/，並輸出
     manifest CSV 記錄每一張圖的搬動紀錄，之後要復原可以直接用備份夾覆蓋回去。
"""

import json
import shutil
import random
import csv
from pathlib import Path
from collections import defaultdict

from config_classification import config
from audit_image_quality import corner_padding_score, TINY_CROP_MAX_SIDE
from PIL import Image

SEED = config.SEED  # 42，跟訓練用的 seed 一致，確保這次切分可重現

SPLITS = [
    ("train", config.TRAIN_DIR, "train_annotations.json"),
    ("val",   config.VAL_DIR,   "val_annotations.json"),
    ("test",  config.TEST_DIR,  "test_annotations.json"),
]
SPLIT_DIR = {s: d for s, d, _ in SPLITS}


def is_flagged(img_path: Path) -> bool:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if max(w, h) < TINY_CROP_MAX_SIDE:
        return True
    padded, _, _ = corner_padding_score(img)
    return padded


def largest_remainder_alloc(total, ratios):
    # 把 total 筆依 ratios 比例分配成整數，總和一定等於 total（避免四捨五入誤差）
    raw = [total * r for r in ratios]
    base = [int(x) for x in raw]
    remainder = total - sum(base)
    order = sorted(range(len(ratios)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in order[:remainder]:
        base[i] += 1
    return base


def main():
    backup_dir = config.DATA_DIR.parent / "merged_data_backup_pre_resplit"
    if backup_dir.exists():
        print(f"備份已存在，略過備份：{backup_dir}")
    else:
        print(f"備份 {config.DATA_DIR} -> {backup_dir} ...")
        shutil.copytree(config.DATA_DIR, backup_dir)
        print("✓ 備份完成")

    # ---- 讀取全部樣本，計算品質旗標 ----
    records = []
    for split, data_dir, ann_file in SPLITS:
        with open(data_dir / ann_file, "r") as f:
            anns = json.load(f)
        for a in anns:
            img_path = data_dir / a["image_path"]
            flagged = is_flagged(img_path)
            records.append({
                "image_path": a["image_path"],
                "label": a["label"],
                "class_name": a["class_name"],
                "image_size": a["image_size"],
                "old_split": split,
                "flagged": flagged,
            })
    print(f"共 {len(records)} 筆樣本，品質旗標計算完成")

    # ---- 依類別重新分配 split，保持每類別原本的 train/val/test 數量不變 ----
    by_class = defaultdict(list)
    for r in records:
        by_class[r["class_name"]].append(r)

    rng = random.Random(SEED)
    new_records = []

    print(f"\n{'class':30s} {'train':>6s} {'val':>6s} {'test':>6s}   "
          f"{'flag%(old T/V/Te)':>20s} -> {'flag%(new T/V/Te)':>20s}")

    for cls, items in sorted(by_class.items()):
        target_n = {"train": 0, "val": 0, "test": 0}
        for r in items:
            target_n[r["old_split"]] += 1

        flagged_pool = [r for r in items if r["flagged"]]
        clean_pool   = [r for r in items if not r["flagged"]]
        rng.shuffle(flagged_pool)
        rng.shuffle(clean_pool)

        splits_order = ["train", "val", "test"]
        total_n = sum(target_n[s] for s in splits_order)
        ratios  = [target_n[s] / total_n for s in splits_order]

        flagged_alloc = largest_remainder_alloc(len(flagged_pool), ratios)
        clean_alloc   = largest_remainder_alloc(len(clean_pool), ratios)

        old_flag_rate, new_flag_rate = {}, {}
        fi, ci = 0, 0
        for s, n_flag, n_clean in zip(splits_order, flagged_alloc, clean_alloc):
            chunk = flagged_pool[fi:fi + n_flag] + clean_pool[ci:ci + n_clean]
            fi += n_flag
            ci += n_clean
            old_n = target_n[s]
            old_flagged_n = sum(1 for r in items if r["old_split"] == s and r["flagged"])
            old_flag_rate[s] = 100 * old_flagged_n / old_n if old_n else 0.0
            new_flag_rate[s] = 100 * n_flag / (n_flag + n_clean) if (n_flag + n_clean) else 0.0
            for r in chunk:
                new_records.append({**r, "new_split": s})

        print(f"{cls:30s} {target_n['train']:6d} {target_n['val']:6d} {target_n['test']:6d}   "
              f"{old_flag_rate['train']:5.1f}/{old_flag_rate['val']:5.1f}/{old_flag_rate['test']:5.1f}"
              f"      -> {new_flag_rate['train']:5.1f}/{new_flag_rate['val']:5.1f}/{new_flag_rate['test']:5.1f}")

    # ---- 搬動實際需要換 split 的檔案 ----
    moved, kept = 0, 0
    manifest = []
    for r in new_records:
        old_dir = SPLIT_DIR[r["old_split"]]
        new_dir = SPLIT_DIR[r["new_split"]]
        manifest.append({
            "image_path": r["image_path"], "class_name": r["class_name"],
            "old_split": r["old_split"], "new_split": r["new_split"],
            "flagged": r["flagged"],
        })
        if r["old_split"] == r["new_split"]:
            kept += 1
            continue
        src = old_dir / r["image_path"]
        dst = new_dir / r["image_path"]
        shutil.move(str(src), str(dst))
        moved += 1

    print(f"\n✓ 搬動 {moved} 張圖片、{kept} 張留在原 split")

    # ---- 重寫三個 annotation json ----
    new_by_split = defaultdict(list)
    for r in new_records:
        new_by_split[r["new_split"]].append({
            "image_path": r["image_path"],
            "label": r["label"],
            "class_name": r["class_name"],
            "image_size": r["image_size"],
        })

    for split, data_dir, ann_file in SPLITS:
        with open(data_dir / ann_file, "w") as f:
            json.dump(new_by_split[split], f, indent=2, ensure_ascii=False)
        print(f"✓ 已更新 {data_dir / ann_file}  ({len(new_by_split[split])} 筆)")

    # ---- manifest ----
    manifest_path = config.RESULTS_DIR / "resplit_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"✓ 搬動紀錄存至 {manifest_path}")
    print(f"✓ 原始資料完整備份在 {backup_dir}（如需復原，用這份覆蓋回 merged_data 即可）")


if __name__ == "__main__":
    main()

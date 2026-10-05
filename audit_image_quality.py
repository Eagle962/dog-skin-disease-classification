"""
資料品質稽核：抓出「四角有黑色三角填充」（旋轉增強殘留）與「裁切過小」的圖片。

背景：Grad-CAM 診斷（gradcam_dermatophytosis.py）顯示 dermatophytosis 誤判最嚴重的
樣本裡，有大量圖片是傾斜嵌在黑色背景中的畫面 —— 所有檔名都帶 Roboflow 匯出格式的
`_jpg.rf.<hash>` 後綴，研判是 Roboflow 專案裡「旋轉增強」後的版本被當成一般圖片，
連同重新計算過的 bounding box 一起被 merge_datasets.py 收進資料集，而不是乾淨原圖。

這支腳本掃過 merged_data 三個 split 的全部樣本（不只 dermatophytosis），用兩個純像素
啟發式規則標記可疑圖片，輸出清單 + 每類別統計 + 一張 contact sheet 讓人工確認，
不會刪除或修改任何原始檔案。
"""

import json
import csv
from pathlib import Path

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_classification import config

SPLITS = [
    ("train", config.TRAIN_DIR, "train_annotations.json"),
    ("val",   config.VAL_DIR,   "val_annotations.json"),
    ("test",  config.TEST_DIR,  "test_annotations.json"),
]

TINY_CROP_MAX_SIDE = 64      # 裁切邊長小於這個值，放大到 224 後會嚴重失真/失去細節
CORNER_DARK_THRESH = 15      # 角落色塊平均亮度低於此值視為「近乎純黑」
CENTER_MIN_MEAN    = 35      # 中心必須夠亮，才能跟「本來就是暗色照片」區分開
MIN_CONTRAST       = 25      # 中心亮度 - 角落亮度，至少要差這麼多才算是黑邊填充


def corner_padding_score(img: Image.Image):
    """回傳 (是否判定為旋轉黑邊, 最暗角落的平均灰階值, 中心區平均灰階值)。"""
    gray = np.array(img.convert("L"), dtype=np.float32)
    h, w = gray.shape
    s = max(4, int(0.12 * min(h, w)))

    corners = [
        gray[0:s, 0:s], gray[0:s, w - s:w],
        gray[h - s:h, 0:s], gray[h - s:h, w - s:w],
    ]
    corner_means = [c.mean() for c in corners]
    darkest = min(corner_means)
    n_dark_corners = sum(1 for m in corner_means if m < CORNER_DARK_THRESH)

    cs = max(4, int(0.2 * min(h, w)))
    cy, cx = h // 2, w // 2
    center = gray[cy - cs:cy + cs, cx - cs:cx + cs]
    center_mean = center.mean()

    is_padded = (
        n_dark_corners >= 2
        and center_mean > CENTER_MIN_MEAN
        and (center_mean - darkest) > MIN_CONTRAST
    )
    return is_padded, darkest, center_mean


def main():
    all_records = []
    for split, data_dir, ann_file in SPLITS:
        with open(data_dir / ann_file, "r") as f:
            anns = json.load(f)
        for a in anns:
            all_records.append({**a, "split": split, "data_dir": data_dir})

    print(f"共 {len(all_records)} 筆樣本，開始稽核...\n")

    flagged = []
    per_class_total  = {}
    per_class_tiny   = {}
    per_class_padded = {}

    for rec in all_records:
        cls = rec["class_name"]
        per_class_total[cls] = per_class_total.get(cls, 0) + 1

        img_path = rec["data_dir"] / rec["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  warn: 無法開啟 {img_path.name}: {e}")
            continue

        w, h = img.size
        is_tiny = max(w, h) < TINY_CROP_MAX_SIDE

        is_padded, darkest, center_mean = corner_padding_score(img)

        if is_tiny:
            per_class_tiny[cls] = per_class_tiny.get(cls, 0) + 1
        if is_padded:
            per_class_padded[cls] = per_class_padded.get(cls, 0) + 1

        if is_tiny or is_padded:
            flagged.append({
                "split": rec["split"],
                "class_name": cls,
                "image_path": rec["image_path"],
                "size": f"{w}x{h}",
                "tiny_crop": is_tiny,
                "rotated_padding": is_padded,
                "corner_darkest": round(darkest, 1),
                "center_mean": round(center_mean, 1),
            })

    # ---- summary table ----
    print(f"{'class':30s} {'total':>6s} {'tiny_crop':>10s} {'rot_padding':>12s}")
    for cls in sorted(per_class_total):
        t = per_class_total[cls]
        tiny = per_class_tiny.get(cls, 0)
        pad  = per_class_padded.get(cls, 0)
        print(f"{cls:30s} {t:6d} {tiny:9d} ({100*tiny/t:4.1f}%) "
              f"{pad:9d} ({100*pad/t:4.1f}%)")

    # ---- CSV ----
    out_csv = config.RESULTS_DIR / "flagged_image_quality.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flagged[0].keys()) if flagged else [])
        if flagged:
            writer.writeheader()
            writer.writerows(flagged)
    print(f"\n✓ 已輸出 {len(flagged)} 筆可疑樣本清單至 {out_csv}")

    # ---- contact sheet：只看 dermatophytosis 裡黑邊分數最嚴重的前 20 張 ----
    derm_padded = sorted(
        [r for r in flagged if r["class_name"] == "dermatophytosis" and r["rotated_padding"]],
        key=lambda r: r["corner_darkest"],
    )[:20]

    if derm_padded:
        cols = 5
        rows = (len(derm_padded) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9))
        axes = np.array(axes).reshape(-1)
        split_dir = {s: d for s, d, _ in SPLITS}
        for i, rec in enumerate(derm_padded):
            img = Image.open(split_dir[rec["split"]] / rec["image_path"]).convert("RGB")
            axes[i].imshow(img)
            axes[i].axis("off")
            axes[i].set_title(f"{rec['split']}\ncorner={rec['corner_darkest']:.0f}", fontsize=8)
        for j in range(len(derm_padded), len(axes)):
            axes[j].axis("off")
        fig.suptitle("dermatophytosis: worst rotated-padding candidates", fontsize=13, fontweight="bold")
        fig.tight_layout()
        out_png = config.RESULTS_DIR / "flagged_dermatophytosis_contact_sheet.png"
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ dermatophytosis 黑邊 contact sheet 存至 {out_png}")
    else:
        print("dermatophytosis 沒有被標記為 rotated_padding 的樣本。")


if __name__ == "__main__":
    main()

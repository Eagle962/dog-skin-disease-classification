import os
import json
import shutil
import hashlib
import yaml
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
from tqdm import tqdm

try:
    from roboflow import Roboflow
except ImportError:
    os.system("pip install roboflow --break-system-packages -q")
    from roboflow import Roboflow

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 類別名稱對照表，把各資料集不同的命名統一
CLASS_ALIAS = {
    "bacterial_pyoderma": "bacterial_pyoderma",
    "bacterial pyoderma": "bacterial_pyoderma",
    "pyoderma":           "bacterial_pyoderma",
    "bacterial infection":"bacterial_pyoderma",
    "hot_spot":           "bacterial_pyoderma",
    "hot spot":           "bacterial_pyoderma",
    "hotspot":            "bacterial_pyoderma",

    "dermatophytosis":    "dermatophytosis",
    "ringworm":           "dermatophytosis",
    "fungal infection":   "dermatophytosis",
    "fungal_infection":   "dermatophytosis",

    "mange":              "mange",
    "scabies":            "mange",
    "sarcoptic_mange":    "mange",
    "sarcoptic mange":    "mange",
    "demodectic_mange":   "mange",
    "demodectic mange":   "mange",
    "demodicosis":        "mange",
    "demodecosis":        "mange",

    "flea_allergy":               "flea_allergy_dermatitis",
    "flea allergy":               "flea_allergy_dermatitis",
    "flea_allergy_dermatitis":    "flea_allergy_dermatitis",
    "flea allergy dermatitis":    "flea_allergy_dermatitis",
    "flea":                       "flea_allergy_dermatitis",

    "allergic_dermatitis":        "allergic_dermatitis",
    "allergic dermatitis":        "allergic_dermatitis",
    "atopic_dermatitis":          "allergic_dermatitis",
    "atopic dermatitis":          "allergic_dermatitis",
    "atopy":                      "allergic_dermatitis",
    "environmental_allergy":      "allergic_dermatitis",
    "food_allergy":               "allergic_dermatitis",

    "seborrhea":                  "seborrhea",
    "seborrheic_dermatitis":      "seborrhea",
    "seborrheic dermatitis":      "seborrhea",
    "seborrhoea":                 "seborrhea",

    "hypersensitivity":           "hypersensitivity",
    "contact_dermatitis":         "hypersensitivity",
    "contact dermatitis":         "hypersensitivity",

    "healthy":    "healthy",
    "normal":     "healthy",
    "no_disease": "healthy",
    "no disease": "healthy",
    "fine":       "healthy",

    "alopecia":   "alopecia",
    "hair_loss":  "alopecia",
    "hair loss":  "alopecia",

    "tumor":      "tumor",
    "neoplasia":  "tumor",
    "mass":       "tumor",
    "neoplasm":   "tumor",

    "intertrigo":            "intertrigo",
    "skin_fold_dermatitis":  "intertrigo",

    "acral_lick_dermatitis": "acral_lick_dermatitis",
    "acral lick dermatitis": "acral_lick_dermatitis",
    "lick_granuloma":        "acral_lick_dermatitis",
}

# "crop"  → 每個 bounding box 裁切成獨立分類樣本（資料量較多）
# "whole" → 整張圖取主要類別（簡單但資訊少）
YOLO_STRATEGY = "crop"
CROP_PADDING  = 0.15   # 裁切時往外擴的比例，讓模型看到更多脈絡

OUTPUT_DIR  = Path("merged_data")
SPLIT_RATIO = (0.75, 0.15, 0.10)

DATASETS = [
    {
        "name":      "litespy_detection",
        "workspace": "litespy-l22hu",
        "project":   "dog-skin-diseases",
        "version":   1,
        "format":    "yolov11",
    },
    {
        "name":      "smartdog_classification",
        "workspace": "smartdog-f8su3",
        "project":   "dog-skin-disease-dataset-bzhug",
        "version":   1,
        "format":    "folder",
    },
    {
        "name":      "research_detection",
        "workspace": "research-on-health",
        "project":   "gabungan-of798",
        "version":   1,
        "format":    "yolov11",
    },
]

API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")

# ---------- utils ----------

_warned_classes: set = set()

def normalize_class_name(raw: str) -> str:
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if key in CLASS_ALIAS:
        return CLASS_ALIAS[key]
    if key not in _warned_classes:
        print(f"  [new class] '{raw}' → '{key}'")
        _warned_classes.add(key)
    return key


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def safe_crop(img: Image.Image, box, padding=CROP_PADDING):
    w, h = img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - bw * padding)
    y1 = max(0, y1 - bh * padding)
    x2 = min(w, x2 + bw * padding)
    y2 = min(h, y2 + bh * padding)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return img.crop((x1, y1, x2, y2))


# ---------- download ----------

def download_dataset(ds_info: dict, api_key: str):
    name      = ds_info["name"]
    fmt       = ds_info["format"]
    dl_format = "folder" if fmt == "folder" else "yolov8"

    print(f"\n{'='*60}")
    print(f"Downloading: {name}  ({fmt})")
    print(f"{'='*60}")

    rf      = Roboflow(api_key=api_key)
    project = rf.workspace(ds_info["workspace"]).project(ds_info["project"])

    # 先試指定版本，找不到就從 1 往上試到 10
    specified = ds_info["version"]
    to_try    = [specified] + [v for v in range(1, 11) if v != specified]

    for ver in to_try:
        try:
            print(f"  trying version {ver}...")
            dataset = project.version(ver).download(dl_format)
            path    = Path(dataset.location)
            print(f"  ✓ version {ver} → {path}")
            return path
        except Exception as e:
            msg = str(e).lower()
            if "not found" in msg or "version number" in msg:
                continue
            print(f"  ✗ failed (v{ver}): {e}")
            return None

    print(f"  ✗ no available version found (tried 1–10)")
    return None


# ---------- parsers ----------

def parse_folder_dataset(dataset_path: Path) -> list:
    records = []
    for split_dir in dataset_path.iterdir():
        if not split_dir.is_dir():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            for img in class_dir.iterdir():
                if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    records.append({
                        "img_path":  img,
                        "class_raw": class_dir.name,
                        "split":     split_dir.name,
                    })
    print(f"  folder: {len(records)} images found")
    return records


def parse_yolo_dataset(dataset_path: Path) -> list:
    yaml_files = list(dataset_path.glob("data.yaml")) + list(dataset_path.glob("data.yml"))
    class_names_yolo = []
    if yaml_files:
        with open(yaml_files[0], "r") as f:
            cfg = yaml.safe_load(f)
        class_names_yolo = cfg.get("names", [])
        print(f"  YOLO classes: {class_names_yolo}")
    else:
        print("  data.yaml not found, class names will be inferred")

    records = []
    for split_dir in dataset_path.iterdir():
        if not split_dir.is_dir():
            continue
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not images_dir.exists():
            continue

        for img in images_dir.iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            label_file = labels_dir / (img.stem + ".txt")
            boxes, class_ids = [], []
            if label_file.exists():
                try:
                    with open(label_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            cid = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:5])
                            boxes.append((cx, cy, bw, bh))
                            class_ids.append(cid)
                except Exception:
                    pass
            records.append({
                "img_path":        img,
                "boxes":           boxes,
                "class_ids":       class_ids,
                "split":           split_dir.name,
                "class_names_yolo": class_names_yolo,
            })

    print(f"  YOLO: {len(records)} images found")
    return records


# ---------- processors ----------

def process_folder_records(records, output_dir, seen_hashes, all_samples):
    skipped = 0
    for rec in tqdm(records, desc="  processing folder"):
        split = "val" if rec["split"] == "valid" else rec["split"]
        if split not in ("train", "val", "test"):
            split = "train"

        try:
            img = Image.open(rec["img_path"]).convert("RGB")
        except Exception as e:
            print(f"  warn: {rec['img_path'].name}: {e}")
            continue

        h = file_hash(rec["img_path"])
        if h in seen_hashes:
            skipped += 1
            continue
        seen_hashes.add(h)

        out_dir = output_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        new_name = f"{h[:8]}_{rec['img_path'].stem}.jpg"
        img.save(out_dir / new_name, "JPEG", quality=95)

        all_samples[split].append({
            "image_path":  new_name,
            "class_name":  normalize_class_name(rec["class_raw"]),
            "image_size":  list(img.size),
        })

    if skipped:
        print(f"  duplicates removed: {skipped}")


def process_yolo_records(records, output_dir, seen_hashes, all_samples,
                         strategy=YOLO_STRATEGY):
    dup = no_label = small = 0

    for rec in tqdm(records, desc="  processing YOLO"):
        split = "val" if rec["split"] == "valid" else rec["split"]
        if split not in ("train", "val", "test"):
            split = "train"

        if not rec["boxes"]:
            no_label += 1
            continue

        try:
            img = Image.open(rec["img_path"]).convert("RGB")
        except Exception as e:
            print(f"  warn: {rec['img_path'].name}: {e}")
            continue

        iw, ih   = img.size
        out_dir  = output_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        yolo_cls = rec["class_names_yolo"]

        if strategy == "crop":
            for i, (cid, (cx, cy, bw, bh)) in enumerate(zip(rec["class_ids"], rec["boxes"])):
                class_raw = yolo_cls[cid] if yolo_cls and cid < len(yolo_cls) else f"class_{cid}"
                x1 = (cx - bw / 2) * iw
                y1 = (cy - bh / 2) * ih
                x2 = (cx + bw / 2) * iw
                y2 = (cy + bh / 2) * ih

                cropped = safe_crop(img, [x1, y1, x2, y2])
                if cropped is None:
                    small += 1
                    continue

                ch = hashlib.md5(cropped.tobytes()).hexdigest()
                if ch in seen_hashes:
                    dup += 1
                    continue
                seen_hashes.add(ch)

                new_name = f"{ch[:8]}_{rec['img_path'].stem}_crop{i}.jpg"
                cropped.save(out_dir / new_name, "JPEG", quality=95)
                all_samples[split].append({
                    "image_path": new_name,
                    "class_name": normalize_class_name(class_raw),
                    "image_size": list(cropped.size),
                })

        else:  # whole
            most_cid  = Counter(rec["class_ids"]).most_common(1)[0][0]
            class_raw = yolo_cls[most_cid] if yolo_cls and most_cid < len(yolo_cls) else f"class_{most_cid}"
            h = file_hash(rec["img_path"])
            if h in seen_hashes:
                dup += 1
                continue
            seen_hashes.add(h)
            new_name = f"{h[:8]}_{rec['img_path'].stem}.jpg"
            img.save(out_dir / new_name, "JPEG", quality=95)
            all_samples[split].append({
                "image_path": new_name,
                "class_name": normalize_class_name(class_raw),
                "image_size": list(img.size),
            })

    if dup:      print(f"  duplicates removed: {dup}")
    if no_label: print(f"  skipped (no label): {no_label}")
    if small:    print(f"  skipped (too small): {small}")


# ---------- finalize ----------

def finalize(all_samples: dict, output_dir: Path):
    all_names = sorted({s["class_name"] for v in all_samples.values() for s in v})
    cls2idx   = {n: i for i, n in enumerate(all_names)}
    n_cls     = len(all_names)

    print(f"\nFinal classes ({n_cls}):")
    for i, name in enumerate(all_names):
        total = sum(sum(1 for s in v if s["class_name"] == name) for v in all_samples.values())
        print(f"  {i:2d}  {name:35s}  {total:4d} samples")

    for split, samples in all_samples.items():
        if not samples:
            continue
        out = output_dir / split
        out.mkdir(parents=True, exist_ok=True)

        anns = [{
            "image_path": s["image_path"],
            "label":      cls2idx[s["class_name"]],
            "class_name": s["class_name"],
            "image_size": s["image_size"],
        } for s in samples]

        with open(out / f"{split}_annotations.json", "w", encoding="utf-8") as f:
            json.dump(anns, f, indent=2, ensure_ascii=False)

        cnt = Counter(a["class_name"] for a in anns)
        print(f"\n{split.upper()} ({len(anns)} samples):")
        for name in all_names:
            if cnt[name] > 0:
                print(f"  {name:35s}: {cnt[name]:4d}")

    _save_config(all_names, cls2idx)
    _plot_distribution(all_samples, all_names, output_dir)


def _save_config(class_names: list, class_to_idx: dict):
    content = f"""import torch
from pathlib import Path


class Config:
    PROJECT_NAME = "Dog_Skin_Disease_Classification"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4
    SEED = 42

    BASE_DIR  = Path(__file__).parent
    DATA_DIR  = BASE_DIR / "merged_data"
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR   = DATA_DIR / "val"
    TEST_DIR  = DATA_DIR / "test"

    OUTPUT_DIR  = BASE_DIR / "outputs"
    WEIGHTS_DIR = OUTPUT_DIR / "weights"
    LOGS_DIR    = OUTPUT_DIR / "logs"
    RESULTS_DIR = OUTPUT_DIR / "results"

    NUM_CLASSES  = {len(class_names)}
    CLASS_NAMES  = {class_names}
    CLASS_TO_IDX = {class_to_idx}

    # efficientnet_b0 在中小型資料集通常比 resnet50 穩
    BACKBONE   = "efficientnet_b0"
    PRETRAINED = True
    IMG_SIZE   = 224

    BATCH_SIZE    = 32
    NUM_EPOCHS    = 60
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY  = 1e-4

    RANDOM_FLIP     = 0.5
    RANDOM_ROTATION = 15
    COLOR_JITTER    = True
    RANDOM_CROP     = True

    LR_SCHEDULER   = "cosine"
    WARMUP_EPOCHS  = 3
    MIN_LR         = 1e-6

    EVAL_METRICS = ["accuracy", "precision", "recall", "f1_score"]

    SAVE_CHECKPOINT_FREQ    = 5
    KEEP_BEST_N             = 3
    EARLY_STOPPING_PATIENCE = 15

    TENSORBOARD_LOG = False

    @classmethod
    def create_directories(cls):
        for d in [cls.DATA_DIR, cls.OUTPUT_DIR, cls.WEIGHTS_DIR,
                  cls.LOGS_DIR, cls.RESULTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)


config = Config()
config.create_directories()
"""
    with open("config_classification.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("\n✓ config_classification.py saved")


def _plot_distribution(all_samples: dict, class_names: list, output_dir: Path):
    splits = [s for s in ("train", "val", "test") if all_samples.get(s)]
    counts = {sp: Counter(s["class_name"] for s in all_samples[sp]) for sp in splits}
    total  = Counter()
    for c in counts.values():
        total.update(c)

    n = len(class_names)

    # total bar chart
    fig, ax = plt.subplots(figsize=(max(10, n), 6))
    bars = ax.bar(range(n), [total[c] for c in class_names], color="#4C72B0", edgecolor="white")
    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Sample Count")
    ax.set_title("Class Distribution (Total)")
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_ylim(0, max(total.values()) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution_total.png", dpi=150)
    plt.close()
    print(f"✓ class_distribution_total.png saved")

    # stacked by split
    colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
    fig, ax = plt.subplots(figsize=(max(10, n), 6))
    bottom = [0] * n
    for sp in splits:
        vals = [counts[sp][c] for c in class_names]
        ax.bar(range(n), vals, bottom=bottom, label=sp,
               color=colors.get(sp, "gray"), edgecolor="white")
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(range(n))
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Sample Count")
    ax.set_title("Class Distribution (by Split)")
    ax.legend(); ax.set_ylim(0, max(bottom) * 1.15); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution_by_split.png", dpi=150)
    plt.close()
    print(f"✓ class_distribution_by_split.png saved")


# ---------- main ----------

def main():
    print("=" * 60)
    print("Dog Skin Disease Dataset Merger")
    print(f"YOLO strategy: {YOLO_STRATEGY}")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_samples  = defaultdict(list)
    seen_hashes: set = set()

    for ds_info in DATASETS:
        path = download_dataset(ds_info, API_KEY)
        if path is None:
            print(f"  skipping {ds_info['name']}")
            continue

        fmt = ds_info["format"]
        if fmt == "folder":
            process_folder_records(
                parse_folder_dataset(path), OUTPUT_DIR, seen_hashes, all_samples)
        elif fmt in ("yolov11", "yolov8"):
            process_yolo_records(
                parse_yolo_dataset(path), OUTPUT_DIR, seen_hashes, all_samples)
        else:
            print(f"  unsupported format: {fmt}, skipped")

    if not any(all_samples.values()):
        print("\nNo samples collected. Check downloads.")
        return

    finalize(all_samples, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Data saved to: {OUTPUT_DIR.resolve()}")
    print("\nNext step:")
    print("  python train_classification.py")


if __name__ == "__main__":
    main()

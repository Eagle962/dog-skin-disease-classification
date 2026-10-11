"""從本機的 CC BY / CC0 來源重建資料集，並以「來源群組」切分。

來源與授權見 dataset_sources.md。不使用已刪除的 Skin-Disease-of-Dog（授權未確認）。

流程：
1. 讀取各來源（YOLO 偵測 -> 依標註框裁切；資料夾分類 -> 整張），統一為 7 類。
2. 裁切後先存成 JPEG bytes，之後所有雜湊都在 JPEG 解碼後的影像上計算（與磁碟檔一致）。
3. 去重：完全相同(md5)；後面來源與前面來源的近重複（pHash<=6 且像素 r>=0.9）丟棄；
   標示 dedup_rot 的來源（Roboflow 旋轉擴增）在來源內也去除（含 90 度旋轉比對）。
4. 剔除旋轉黑邊樣本（has_black_border）。
5. 近重複/同一來源照片 -> 同一群組 -> 整組分到同一 split；各類別依比例分層。
"""
import csv
import hashlib
import io
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import imagehash
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).parent
OUT = ROOT / "dataset_cc_by"
SEED = 42
CROP_PADDING = 0.15
PHASH_T = 6
R_T = 0.9
SD_MIN = 12  # 標準差過低(近乎單色)的影像 pHash 不可靠，不參與近重複判定
# 目標比例 = 5796 : 724 : 544
RATIO = {"train": 5796 / 7064, "val": 724 / 7064, "test": 544 / 7064}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SOURCES = [  # 順序即去重優先序（前面的保留）
    {"name": "litespy", "dir": "Dog-Skin-Diseases-1", "kind": "yolo"},
    {"name": "smartdog", "dir": "Dog-Skin-Disease-Dataset-1", "kind": "folder"},
    {"name": "gabungan", "dir": "gabungan-1", "kind": "yolo"},
    # 新增來源需通過人工目視品質檢查；最小邊長過小的裁切丟棄。
    # 已評估但不採用：pethealth（與 litespy 同類的近單色色偏圖，近重複無法可靠判定）、
    # yashmotiani（65% 與現有資料重複，其餘混有色偏圖與旋轉擴增圖）。
    {"name": "smoky", "dir": "candidates/rf_smoky", "kind": "yolo", "min_side": 64},
]

# None = 不屬於 7 類，丟棄
ALIAS = {
    "bacterial_dermatosis": "bacterial_dermatosis",
    "dermatitis": "dermatitis",
    "fungal_infection": "dermatophytosis",
    "fungal_infections": "dermatophytosis",
    "ringworm": "dermatophytosis",
    "dermatophytosis": "dermatophytosis",
    "flea_allergy": "flea_allergy_dermatitis",
    "healthy": "healthy",
    "hypersensitivity_dermatitis": "hypersensitivity_dermatitis",
    "hypersensitivity_allergic_dermatosis": "hypersensitivity_dermatitis",
    "scabies": "mange",
    "mange": "mange",
    "tumor": None,
}


def norm(raw):
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in ALIAS:
        raise KeyError(f"未對映的類別: {raw!r}")
    return ALIAS[key]


def source_photo_id(name):
    return re.split(r"\.rf\.", Path(name).stem)[0].lower()


def crop(img, cx, cy, bw, bh):
    iw, ih = img.size
    x1, y1 = (cx - bw / 2) * iw, (cy - bh / 2) * ih
    x2, y2 = (cx + bw / 2) * iw, (cy + bh / 2) * ih
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - w * CROP_PADDING), max(0, y1 - h * CROP_PADDING)
    x2, y2 = min(iw, x2 + w * CROP_PADDING), min(ih, y2 + h * CROP_PADDING)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return img.crop((x1, y1, x2, y2))


# 旋轉擴增殘留的黑色三角邊（準則同 audit_image_quality.py，已目視確認無誤判）
CORNER_DARK_THRESH, CENTER_MIN_MEAN, MIN_CONTRAST = 15, 35, 25


def has_black_border(img):
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = gray.shape
    k = max(4, int(0.12 * min(h, w)))
    corners = [gray[:k, :k], gray[:k, w - k:], gray[h - k:, :k], gray[h - k:, w - k:]]
    means = [c.mean() for c in corners]
    cs = max(4, int(0.2 * min(h, w)))
    center = gray[h // 2 - cs:h // 2 + cs, w // 2 - cs:w // 2 + cs].mean()
    return (sum(m < CORNER_DARK_THRESH for m in means) >= 2
            and center > CENTER_MIN_MEAN and center - min(means) > MIN_CONTRAST)


def to_jpeg(im):
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=95)
    data = buf.getvalue()
    return data, Image.open(io.BytesIO(data)).convert("RGB")


def collect():
    seen, rows, stats = set(), [], Counter()
    for src in SOURCES:
        d = ROOT / src["dir"]
        names = yaml.safe_load(open(d / "data.yaml"))["names"] if src["kind"] == "yolo" else None
        for split_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            if src["kind"] == "folder":
                items = [(p, c.name) for c in split_dir.iterdir() if c.is_dir()
                         for p in c.iterdir() if p.suffix.lower() in IMG_EXT]
            else:
                idir = split_dir / "images"
                items = [(p, None) for p in idir.glob("*")
                         if p.suffix.lower() in IMG_EXT] if idir.exists() else []
            for p, cname in sorted(items):
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    stats[src["name"] + ":unreadable"] += 1
                    continue
                if src["kind"] == "folder":
                    pieces = [(img, norm(cname), "")]
                else:
                    lab = d / split_dir.name / "labels" / (p.stem + ".txt")
                    pieces = []
                    if lab.exists():
                        for i, line in enumerate(open(lab)):
                            q = line.split()
                            if len(q) < 5:
                                continue
                            cls = norm(names[int(q[0])])
                            if cls is None:
                                stats[src["name"] + ":dropped_class"] += 1
                                continue
                            c = crop(img, *map(float, q[1:5]))
                            if c is None:
                                stats[src["name"] + ":too_small"] += 1
                                continue
                            pieces.append((c, cls, f"_crop{i}"))
                    if not pieces:
                        stats[src["name"] + ":no_usable_label"] += 1
                for im, cls, suf in pieces:
                    if has_black_border(im):
                        stats[src["name"] + ":black_border"] += 1
                        continue
                    if min(im.size) < src.get("min_side", 0):
                        stats[src["name"] + ":below_min_side"] += 1
                        continue
                    data, im = to_jpeg(im)
                    h = hashlib.md5(data).hexdigest()
                    if h in seen:
                        stats[src["name"] + ":exact_dup"] += 1
                        continue
                    seen.add(h)
                    rows.append({"jpg": data, "img": im, "source": src["name"],
                                 "orig": p.name + suf, "cls": cls, "md5": h,
                                 "photo": f'{src["name"]}/{p.stem.lower()}',
                                 "base": source_photo_id(p.name)})
    return rows, stats


def popcount(x):
    return np.bitwise_count(x).astype(np.int64)


def features(rows):
    """pHash 與 32x32 灰階 z-score；另存 90/180/270 度旋轉版本供旋轉不變比對。"""
    H, P, SD, HR, PR = [], [], [], [], []
    for r in rows:
        im = r["img"]
        g = np.asarray(im.convert("L").resize((32, 32)), dtype=np.float32).ravel()
        SD.append(g.std())
        P.append((g - g.mean()) / max(g.std(), 1e-6))
        H.append(int(str(imagehash.phash(im)), 16))
        hr, pr = [], []
        for ang in (90, 180, 270):
            ri = im.rotate(ang, expand=True)
            gg = np.asarray(ri.convert("L").resize((32, 32)), dtype=np.float32).ravel()
            pr.append((gg - gg.mean()) / max(gg.std(), 1e-6))
            hr.append(int(str(imagehash.phash(ri)), 16))
        HR.append(hr)
        PR.append(pr)
    return (np.array(H, dtype=np.uint64), np.stack(P), np.array(SD),
            np.array(HR, dtype=np.uint64), np.stack(PR))


def match(i, js, F, rot, t=PHASH_T):
    """i 與候選 js 中的近重複（rot=True 時 i 轉 90/180/270 度後也比對）。回傳匹配的 j。"""
    H, P, SD, HR, PR = F
    js = np.asarray(js, dtype=int)
    if SD[i] < SD_MIN or len(js) == 0:
        return np.array([], dtype=int)
    js = js[SD[js] >= SD_MIN]
    ok = np.zeros(len(js), dtype=bool)
    variants = [(H[i], P[i])]
    if rot:
        variants += [(HR[i, k], PR[i, k]) for k in range(3)]
    for h, p in variants:
        idx = np.nonzero(popcount(H[js] ^ h) <= t)[0]
        if len(idx):
            rr = (P[js[idx]] @ p) / P.shape[1]
            ok[idx[rr >= R_T]] = True
    return js[ok]


def drop_dups(rows, F, stats):
    order = {s["name"]: k for k, s in enumerate(SOURCES)}
    rot_src = {s["name"] for s in SOURCES if s.get("dedup_rot")}
    rank = np.array([order[r["source"]] for r in rows])
    keep = np.ones(len(rows), dtype=bool)
    idx_all = np.arange(len(rows))
    for i in np.argsort(rank, kind="stable"):
        src = rows[i]["source"]
        rot = src in rot_src
        earlier = idx_all[keep & (rank < rank[i])]
        if len(match(i, earlier, F, rot)):
            keep[i] = False
            stats[src + ":dup_of_earlier_source"] += 1
            continue
        if rot:  # 來源內的旋轉擴增副本：保留第一個
            same = idx_all[keep & (rank == rank[i]) & (idx_all < i)]
            if len(match(i, same, F, True)):
                keep[i] = False
                stats[src + ":augmented_copy"] += 1
    return keep


def group_ids(rows, F):
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(a)] = find(b)

    by_photo = {}
    for i, r in enumerate(rows):
        union(i, by_photo.setdefault(r["photo"], i))  # 同一來源影像檔的多個裁切
    by_name = {}
    for i, r in enumerate(rows):
        if len(r["base"]) > 20:  # 通用檔名(pic0_jpg 等)不可據以合併
            union(i, by_name.setdefault(r["base"], i))
    for i in range(len(rows)):
        for j in match(i, np.arange(i + 1, len(rows)), F, True, t=PHASH_T + 2):  # 分組較保守
            union(i, int(j))
    return [find(i) for i in range(len(rows))]


def split_groups(rows, gid):
    rng = random.Random(SEED)
    groups = defaultdict(list)
    for i, g in enumerate(gid):
        groups[g].append(i)
    classes = sorted({r["cls"] for r in rows})
    total = Counter(r["cls"] for r in rows)
    target = {s: {c: total[c] * RATIO[s] for c in classes} for s in RATIO}
    cur = {s: Counter() for s in RATIO}
    order = list(groups.values())
    rng.shuffle(order)
    order.sort(key=len, reverse=True)
    assign = {}
    for idxs in order:
        cnt = Counter(rows[i]["cls"] for i in idxs)
        best = max(RATIO, key=lambda s: sum(
            n * (target[s][c] - cur[s][c]) / max(target[s][c], 1) for c, n in cnt.items()))
        for i in idxs:
            assign[i] = best
        cur[best].update(cnt)
    return assign


def main():
    rows, stats = collect()
    print("收集:", len(rows), "樣本")
    F = features(rows)
    keep = drop_dups(rows, F, stats)
    rows = [r for r, k in zip(rows, keep) if k]
    F = features(rows)
    print("去重後:", len(rows), "樣本")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    gid = group_ids(rows, F)
    print("群組數:", len(set(gid)), " 最大群組:", Counter(gid).most_common(1)[0][1])
    assign = split_groups(rows, gid)

    if OUT.exists():
        shutil.rmtree(OUT)
    classes = sorted({r["cls"] for r in rows})
    c2i = {c: i for i, c in enumerate(classes)}
    ann = defaultdict(list)
    manifest = []
    for i, r in enumerate(rows):
        s = assign[i]
        d = OUT / s
        d.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^\w.\-]", "_", Path(r["orig"]).stem)[:60]
        name = f'{r["md5"][:8]}_{r["source"]}_{stem}.jpg'
        (d / name).write_bytes(r["jpg"])
        ann[s].append({"image_path": name, "label": c2i[r["cls"]], "class_name": r["cls"],
                       "image_size": list(r["img"].size), "source": r["source"],
                       "group": int(gid[i])})
        manifest.append([s, name, r["source"], r["orig"], r["cls"], gid[i]])
    for s, a in ann.items():
        json.dump(a, open(OUT / s / f"{s}_annotations.json", "w"), indent=2, ensure_ascii=False)
    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "file", "source", "orig_file", "class", "group"])
        w.writerows(manifest)
    json.dump(classes, open(OUT / "classes.json", "w"), indent=2)

    n = len(rows)
    print(f"\n總數 {n}；目標比例 82.05/10.25/7.70 -> 實際 "
          + "/".join(f"{100 * len(ann[s]) / n:.2f}" for s in RATIO))
    print("類別 x 切分 (train/val/test):")
    for c in classes:
        print(f"  {c:30s}", [sum(1 for a in ann[s] if a["class_name"] == c) for s in RATIO])
    print("  合計", [len(ann[s]) for s in RATIO])
    print("來源 x 切分:")
    for src in SOURCES:
        print(f'  {src["name"]:12s}', [sum(1 for a in ann[s] if a["source"] == src["name"]) for s in RATIO])


if __name__ == "__main__":
    main()

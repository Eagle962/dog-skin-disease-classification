"""以 main.py 的推論路徑（TRANSFORM + 水平翻轉 TTA）對測試集預測，輸出 CSV。
用法：MODEL_PATH=<權重> python verify_deploy_parity.py <test資料夾> <輸出csv>
離線（專案 venv）與 Docker（torch 2.3）各跑一次，再比對兩份 CSV。"""
import csv, json, sys
from pathlib import Path
from PIL import Image
import main

test_dir, out = Path(sys.argv[1]), sys.argv[2]
ann = json.load(open(test_dir / "test_annotations.json"))
ok = 0
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["image", "label"] + [f"p_{c}" for c in main.CLASS_NAMES])
    for a in ann:
        img = Image.open(test_dir / a["image_path"]).convert("RGB")
        p = main.predict_probs(main.TRANSFORM(img).unsqueeze(0).to(main.device))
        ok += int(p.argmax()) == a["label"]
        w.writerow([a["image_path"], a["label"]] + [f"{x:.6f}" for x in p])
print(f"torch {main.torch.__version__}  acc={ok}/{len(ann)}={ok/len(ann):.4f}")

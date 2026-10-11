"""下載 Roboflow 來源資料集到 build_dataset.py 預期的資料夾（已存在則跳過）。

需要 Roboflow API key，依序從以下位置找：
  1. 環境變數 ROBOFLOW_API_KEY
  2. ~/.config/woofai-creds/roboflow.json（JSON {"api_key": "..."}，或含 api_key="..." 的程式碼片段，或單純的 key）
Key 不會被印出或寫入任何檔案。
用法：python download_sources.py
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# (workspace, project, version, 格式, 輸出資料夾)  皆為 CC BY 4.0，詳見 dataset_sources.md
SOURCES = [
    ("litespy-l22hu", "dog-skin-diseases", 1, "yolov8", "Dog-Skin-Diseases-1"),
    ("smartdog-f8su3", "dog-skin-disease-dataset-bzhug", 1, "folder", "Dog-Skin-Disease-Dataset-1"),
    ("research-on-health", "gabungan-of798", 1, "yolov8", "gabungan-1"),
    ("smoky", "dogs-skin-disease-manual", 2, "yolov8", "candidates/rf_smoky"),
]


def find_key():
    k = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if k:
        return k
    f = Path.home() / ".config/woofai-creds/roboflow.json"
    if f.exists():
        s = f.read_text(encoding="utf-8").strip()
        m = re.search(r'api_key\s*[=:]\s*["\']([^"\']+)["\']', s)
        if m:
            return m.group(1)
        try:
            return json.loads(s)["api_key"]
        except Exception:
            if re.fullmatch(r"[A-Za-z0-9]{10,}", s):
                return s
    return ""


def main():
    todo = [s for s in SOURCES if not (ROOT / s[4]).exists()]
    if not todo:
        print("✓ 所有來源資料夾都已存在，不需下載。")
        return
    key = find_key()
    if not key:
        sys.exit("找不到 Roboflow API key（設定 ROBOFLOW_API_KEY 或放入 ~/.config/woofai-creds/roboflow.json）")
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("請先安裝：pip install roboflow")
    rf = Roboflow(api_key=key)
    for ws, pj, ver, fmt, out in todo:
        print(f"下載 {ws}/{pj} v{ver} -> {out}")
        (ROOT / out).parent.mkdir(parents=True, exist_ok=True)
        rf.workspace(ws).project(pj).version(ver).download(fmt, location=str(ROOT / out))
    print("✓ 完成。接著執行：python build_dataset.py")


if __name__ == "__main__":
    main()

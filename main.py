"""
Pet Skin Disease Classifier — FastAPI backend

2026-10-09 修正重點（相對於原版 main.py）：
  1. 權重載入：原版用 raw.get("model", raw.get("state_dict", raw)) 取權重，訓練程式
     存的鍵名其實是 "model_state_dict"，導致取到整個 checkpoint dict；又因為
     strict=False，state_dict 的 311 個張量（213 個可學習參數＋98 個 BatchNorm 統計量）
     全部未載入卻沒有報錯，模型停留在初始化（隨機）權重，所有影像都被判為 mange。
     現改為優先讀 "model_state_dict"，並以 strict=True 載入；載入失敗就直接中止啟動，
     不再默默改用隨機權重。
  2. 權重格式：建議部署 export_weights.py 匯出的精簡權重（只含 state_dict 與少量中繼
     資料，約 16 MB），可用 torch.load(weights_only=True) 安全載入；仍相容完整的
     訓練 checkpoint（best_model.pth）。
  3. 推論與論文評估一致：預設做水平翻轉 TTA（原圖與翻轉圖的 softmax 平均），與
     train_classification.py 最終測試評估的設定相同；可用環境變數 TTA=0 關閉。
  4. Grad-CAM：改用 register_full_backward_hook（原版的 register_backward_hook 已棄用，
     在多運算子模組上可能取到錯誤梯度）。
  5. /health 回報模型是否載入、來源檔與 epoch，方便部署後確認。
  6. 2026-10-12：骨幹與輸入尺寸改由環境變數 WOOF_BACKBONE（efficientnet_b0 / efficientnet_b2，
     預設 efficientnet_b0）與 WOOF_IMG_SIZE（預設 224）控制。CC BY 資料集重訓結果 B0@224 測試
     Accuracy 80.16% 優於 B2@256 的 74.73%，故預設為 B0；換骨幹時兩個變數必須與權重一致。
  /predict 的回傳格式（predictions / top_class / gradcam_image）維持不變，Android 端不需修改。
"""
import io
import base64
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision import transforms
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import cv2

app = FastAPI(title="Pet Skin Disease Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 類別順序必須與訓練時 config_classification.CLASS_NAMES 完全一致
CLASS_NAMES = [
    "bacterial_dermatosis",
    "dermatitis",
    "dermatophytosis",
    "flea_allergy_dermatitis",
    "healthy",
    "hypersensitivity_dermatitis",
    "mange",
]

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/model.pth"))
USE_TTA = os.getenv("TTA", "1") != "0"
BACKBONE = os.getenv("WOOF_BACKBONE", "efficientnet_b0")
IMG_SIZE = int(os.getenv("WOOF_IMG_SIZE", "224"))
if BACKBONE not in ("efficientnet_b0", "efficientnet_b2"):
    raise RuntimeError(f"不支援的 WOOF_BACKBONE：{BACKBONE}")

# 與 train_classification.py 的 val/test transform 相同
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: Optional[nn.Module] = None
MODEL_INFO = {}


def build_model(num_classes: int) -> nn.Module:
    # 與訓練時相同的架構；Dropout 機率不影響推論，參數鍵名為 classifier.1.weight / bias
    m = getattr(tv_models, BACKBONE)(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(m.classifier[1].in_features, num_classes),
    )
    return m


def _read_checkpoint(path: Path):
    try:
        # 精簡權重（export_weights.py 匯出）只含張量與基本型別，可安全以 weights_only 載入
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        # 完整訓練 checkpoint 含 optimizer/scheduler 狀態與 numpy 物件，需關閉 weights_only。
        # 只對自己訓練產生、可信任的檔案這樣做。
        return torch.load(path, map_location=device, weights_only=False)


def load_model():
    global model, MODEL_INFO
    if not MODEL_PATH.exists():
        raise RuntimeError(f"找不到模型檔：{MODEL_PATH}（請確認 docker-compose.yml 的掛載路徑）")

    raw = _read_checkpoint(MODEL_PATH)
    if isinstance(raw, nn.Module):
        raise RuntimeError("模型檔是完整 nn.Module 物件；請改用 state_dict 或 export_weights.py 匯出的權重")
    if not isinstance(raw, dict):
        raise RuntimeError(f"無法辨識的模型檔格式：{type(raw)}")

    if "model_state_dict" in raw:          # train_classification.py 的 checkpoint 與精簡權重
        state_dict = raw["model_state_dict"]
    elif "state_dict" in raw:
        state_dict = raw["state_dict"]
    elif "model" in raw and isinstance(raw["model"], dict):
        state_dict = raw["model"]
    else:                                   # 純 state_dict
        state_dict = raw

    cleaned = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    m = build_model(len(CLASS_NAMES))
    m.load_state_dict(cleaned, strict=True)   # 任何鍵不符都會直接報錯，避免默默使用隨機權重
    model = m.to(device).eval()

    names = raw.get("class_names")
    if names is not None and list(names) != CLASS_NAMES:
        raise RuntimeError(f"權重檔的類別順序與 CLASS_NAMES 不一致：{names}")

    MODEL_INFO = {
        "model_path": str(MODEL_PATH),
        "source_checkpoint": raw.get("source_checkpoint", MODEL_PATH.name),
        "epoch": raw.get("epoch"),
        "val_accuracy": raw.get("best_acc", raw.get("val_accuracy")),
        "backbone": BACKBONE,
        "img_size": IMG_SIZE,
        "tta": USE_TTA,
    }
    print(f"[INFO] Model loaded: {MODEL_INFO}")


load_model()


# ── Grad-CAM ───────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, mdl, target_layer):
        self.mdl = mdl
        self.gradients = None
        self.activations = None
        self._h = [
            target_layer.register_forward_hook(
                lambda m, i, o: setattr(self, "activations", o.detach())
            ),
            target_layer.register_full_backward_hook(
                lambda m, gi, go: setattr(self, "gradients", go[0].detach())
            ),
        ]

    def generate(self, tensor, class_idx):
        self.mdl.zero_grad()
        out = self.mdl(tensor)
        out[0, class_idx].backward()
        w = self.gradients.mean(dim=(2, 3), keepdim=True)          # α_k^c
        cam = F.relu((w * self.activations).sum(dim=1))[0]          # ReLU(Σ_k α_k^c A^k)
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    def remove(self):
        for h in self._h:
            h.remove()


def gradcam_b64(tensor: torch.Tensor, class_idx: int, orig: Image.Image) -> str:
    gc = GradCAM(model, model.features[-1])
    try:
        cam = gc.generate(tensor, class_idx)
    finally:
        gc.remove()
    cam = cv2.resize(cam, (orig.width, orig.height), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    orig_np = np.array(orig.convert("RGB"))
    overlay = (0.5 * orig_np + 0.5 * heatmap).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ── Schemas ────────────────────────────────────────────────────────────────────

class Prediction(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    predictions: List[Prediction]
    top_class: str
    gradcam_image: Optional[str] = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "model_loaded": model is not None, **MODEL_INFO}


@torch.no_grad()
def predict_probs(t: torch.Tensor) -> np.ndarray:
    probs = F.softmax(model(t), dim=1)
    if USE_TTA:
        probs = (probs + F.softmax(model(torch.flip(t, dims=[3])), dim=1)) / 2
    return probs[0].cpu().numpy()


@app.post("/predict", response_model=PredictResponse)
async def predict(image: UploadFile = File(...)):
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(400, "請上傳圖片檔案")
    raw = await image.read()
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "無法解析圖片")

    t = TRANSFORM(pil).unsqueeze(0).to(device)
    probs = predict_probs(t)

    order = np.argsort(probs)[::-1]
    predictions = [
        Prediction(label=CLASS_NAMES[i], probability=float(round(float(probs[i]), 6)))
        for i in order
    ]
    top_idx = int(order[0])

    cam = None
    try:
        cam = gradcam_b64(t.clone().requires_grad_(True), top_idx, pil)
    except Exception as e:
        print(f"[WARN] Grad-CAM failed: {e}")

    return PredictResponse(
        predictions=predictions,
        top_class=CLASS_NAMES[top_idx],
        gradcam_image=cam,
    )

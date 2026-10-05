"""
推論腳本 - 使用 Grad-CAM 標出病灶位置
用法：
    python3 predict.py path/to/image.jpg
    python3 predict.py path/to/folder/   # 批次處理整個資料夾
"""

import sys
import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    resnet50, ResNet50_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
)
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    from config_classification import config
except ImportError:
    print("找不到 config_classification.py")
    sys.exit(1)


# ---------- 裝置偵測 ----------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()


# ---------- 模型載入 ----------

def build_model(backbone, num_classes):
    if backbone == "efficientnet_b0":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == "resnet50":
        model = resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


def load_model(checkpoint_path):
    model = build_model(config.BACKBONE, config.NUM_CLASSES)
    ckpt  = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE).eval()

    epoch   = ckpt.get("epoch", "?")
    acc     = ckpt.get("best_acc", None)
    acc_str = f"{acc:.4f}" if acc is not None else "?"
    print(f"✓ Model loaded  epoch={epoch}  best_acc={acc_str}  device={DEVICE}")
    return model


# ---------- Grad-CAM ----------

class GradCAM:
    """
    Grad-CAM for EfficientNet / ResNet / MobileNet.
    自動抓最後一個 Conv 層的輸出當 feature map。
    """

    def __init__(self, model, backbone):
        self.model    = model
        self.backbone = backbone
        self._feat    = None
        self._grad    = None
        self._hook_handles = []
        self._register_hooks()

    def _target_layer(self):
        if self.backbone == "efficientnet_b0":
            return self.model.features[-1]
        elif self.backbone == "resnet50":
            return self.model.layer4[-1]
        elif self.backbone == "mobilenet_v3_small":
            return self.model.features[-1]
        raise ValueError(f"Unknown backbone: {self.backbone}")

    def _register_hooks(self):
        layer = self._target_layer()

        def fwd_hook(_, __, output):
            self._feat = output.detach()

        def bwd_hook(_, __, grad_output):
            self._grad = grad_output[0].detach()

        self._hook_handles.append(layer.register_forward_hook(fwd_hook))
        self._hook_handles.append(layer.register_full_backward_hook(bwd_hook))

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()

        # global average pooling over spatial dims
        weights = self._grad.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self._feat).sum(dim=1).squeeze()  # (H, W)
        cam     = torch.relu(cam)

        # normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy()

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()


# ---------- 前處理 ----------

TRANSFORM = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def preprocess(image_path):
    img    = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    return img, tensor


# ---------- 推論 ----------

def predict(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top5_idx  = probs.argsort()[::-1][:5]
    top5      = [(config.CLASS_NAMES[i], float(probs[i])) for i in top5_idx]
    pred_idx  = int(top5_idx[0])
    return pred_idx, top5


# ---------- 視覺化 ----------

def visualize(image_path, pred_idx, top5, cam_map, out_path):
    orig = Image.open(image_path).convert("RGB")
    w, h = orig.size

    # resize CAM to original image size
    cam_resized = Image.fromarray((cam_map * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    cam_arr     = np.array(cam_resized) / 255.0
    heatmap     = cm.jet(cam_arr)[:, :, :3]   # (H, W, 3)  RGB
    orig_arr    = np.array(orig) / 255.0

    overlay = (0.5 * orig_arr + 0.5 * heatmap).clip(0, 1)

    # ---------- layout ----------
    fig = plt.figure(figsize=(14, 5))
    gs  = fig.add_gridspec(1, 3, wspace=0.35)

    # 原圖
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(orig)
    ax0.set_title("Original Image", fontweight="bold")
    ax0.axis("off")

    # Grad-CAM overlay
    ax1 = fig.add_subplot(gs[1])
    ax1.imshow(overlay)
    ax1.set_title(f"Grad-CAM  →  {config.CLASS_NAMES[pred_idx]}", fontweight="bold")
    ax1.axis("off")

    # 信心度bar
    ax2 = fig.add_subplot(gs[2])
    names  = [n for n, _ in top5]
    scores = [s for _, s in top5]
    colors = ["#e74c3c" if i == 0 else "#3498db" for i in range(len(top5))]
    bars   = ax2.barh(names[::-1], scores[::-1], color=colors[::-1])
    ax2.bar_label(bars, labels=[f"{s:.1%}" for s in scores[::-1]],
                  padding=4, fontsize=9)
    ax2.set_xlim(0, 1.15)
    ax2.set_xlabel("Confidence")
    ax2.set_title("Top-5 Predictions", fontweight="bold")
    ax2.grid(axis="x", alpha=0.3)

    
    conf = scores[0]
    if conf >= 0.8:
        level, color = "High confidence", "#27ae60"
    elif conf >= 0.5:
        level, color = "Medium confidence", "#f39c12"
    else:
        level, color = "Low confidence — consult a vet", "#e74c3c"

    fig.suptitle(
        f"Prediction: {config.CLASS_NAMES[pred_idx]}  ({conf:.1%})  [{level}]",
        fontsize=13, fontweight="bold", color=color, y=1.01
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Result saved: {out_path}")



def run_single(model, gradcam, image_path, out_dir):
    image_path = Path(image_path)
    print(f"\nProcessing: {image_path.name}")

    orig, tensor = preprocess(image_path)

    # 要讓 Grad-CAM 做 backward 需要 grad
    tensor.requires_grad_(True)
    pred_idx, top5 = predict(model, tensor)

    # 重新跑一次取 CAM（predict 用 no_grad，這裡要重跑）
    model.zero_grad()
    logits = model(tensor)
    logits[0, pred_idx].backward()
    cam = gradcam.generate(tensor, pred_idx)

    print(f"  Prediction : {config.CLASS_NAMES[pred_idx]}")
    print(f"  Confidence : {top5[0][1]:.1%}")
    print(f"  Top-5      :")
    for name, score in top5:
        print(f"    {name:35s}: {score:.1%}")

    out_path = out_dir / f"result_{image_path.stem}.png"
    visualize(image_path, pred_idx, top5, cam, out_path)
    return pred_idx, top5


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 predict.py image.jpg")
        print("  python3 predict.py folder/")
        sys.exit(1)

    checkpoint = config.WEIGHTS_DIR / "best_model.pth"
    if not checkpoint.exists():
        print(f"Model not found: {checkpoint}")
        print("Train the model first: python3 train_classification.py")
        sys.exit(1)

    if not config.CLASS_NAMES:
        print("CLASS_NAMES is empty in config_classification.py")
        print("Run patch_annotations.py first.")
        sys.exit(1)

    model   = load_model(checkpoint)
    gradcam = GradCAM(model, config.BACKBONE)

    out_dir = config.RESULTS_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    target = Path(sys.argv[1])

    if target.is_dir():
        exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [f for f in target.iterdir() if f.suffix.lower() in exts]
        print(f"Found {len(images)} images in {target}")
        for img_path in images:
            try:
                run_single(model, gradcam, img_path, out_dir)
            except Exception as e:
                print(f"  ✗ Failed: {img_path.name}: {e}")
    else:
        run_single(model, gradcam, target, out_dir)

    gradcam.remove_hooks()
    print(f"\nAll results saved to: {out_dir}")


if __name__ == "__main__":
    main()
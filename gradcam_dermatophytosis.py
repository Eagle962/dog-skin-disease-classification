"""
Grad-CAM 診斷：只看 dermatophytosis 被誤判的樣本，模型注意力落在哪裡。

用途：train_classification.py 訓練出的模型在 dermatophytosis 上 recall 長期偏低
（~20-24%），且訓練策略上的調整（Focal Loss / MixUp / 凍結暖身等）幾乎沒有改善這個
類別，懷疑問題出在資料本身（跨資料集合併時的裁切圖 vs 整圖混雜、domain shift）。
這支腳本挑出模型「最有把握猜錯」的 dermatophytosis 樣本，畫出 Grad-CAM 熱力圖，
用來判斷模型到底有沒有在看病灶，還是在看背景/裁切邊界等無關線索。

不改動模型架構，只載入已訓練好的 best_model.pth 做推論用。
"""

import torch
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_classification import config
from train_classification import create_model, DogSkinClassificationDataset, get_device

CLASS_NAMES  = config.CLASS_NAMES
TARGET_CLASS = "dermatophytosis"
TARGET_IDX   = config.CLASS_TO_IDX[TARGET_CLASS]
TOP_N        = 16  # 挑信心最高的誤判樣本，最能反映系統性問題

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, output):
        self.activations = output

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        out = self.model(x)
        out[:, class_idx].sum().backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 1e-8:
            cam /= cam.max()
        return cam, out.detach()


def denormalize(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    return np.clip(img, 0, 1)


def overlay_heatmap(img, cam, alpha=0.45):
    heatmap = plt.cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * img + alpha * heatmap, 0, 1)


def get_target_layer(model, backbone):
    if backbone in ("efficientnet_b0", "mobilenet_v3_small"):
        return model.features[-1]
    if backbone == "resnet50":
        return model.layer4[-1]
    raise ValueError(f"Unknown backbone: {backbone}")


def main():
    device = get_device()
    print(f"Device: {device}")

    model = create_model(config.NUM_CLASSES, config.BACKBONE, pretrained=False).to(device)
    ckpt_path = config.WEIGHTS_DIR / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✓ Loaded {ckpt_path.name} (epoch {ckpt['epoch']})")

    if (config.TEST_DIR / "test_annotations.json").exists():
        data_dir, mode = config.TEST_DIR, "test"
    else:
        data_dir, mode = config.VAL_DIR, "val"
    dataset = DogSkinClassificationDataset(data_dir, mode=mode)

    cam_engine = GradCAM(model, get_target_layer(model, config.BACKBONE))

    # 第一輪：快速推論找出所有 dermatophytosis 樣本的預測結果（不需要梯度）
    records = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            if label != TARGET_IDX:
                continue
            prob = torch.softmax(model(img.unsqueeze(0).to(device)), dim=1)[0]
            pred = int(prob.argmax())
            records.append({
                "idx": idx, "pred": pred,
                "conf": float(prob[pred]), "correct": pred == label,
            })

    n_total = len(records)
    n_wrong = sum(1 for r in records if not r["correct"])
    recall  = 100 * (n_total - n_wrong) / n_total if n_total else 0.0
    print(f"\ndermatophytosis 樣本共 {n_total} 張，誤判 {n_wrong} 張 (recall={recall:.1f}%)")

    # 依信心值排序，挑「模型最有把握猜錯」的樣本 —— 最能反映系統性偏誤
    wrong = sorted([r for r in records if not r["correct"]], key=lambda r: -r["conf"])[:TOP_N]
    if not wrong:
        print("沒有誤判樣本，不需要 Grad-CAM 診斷。")
        return

    out_dir = config.RESULTS_DIR / "gradcam_dermatophytosis"
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(wrong)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.6))
    axes = np.array(axes).reshape(-1)

    for i, rec in enumerate(wrong):
        img, _ = dataset[rec["idx"]]
        x = img.unsqueeze(0).to(device)
        cam, _ = cam_engine(x, rec["pred"])

        base    = denormalize(img)
        overlay = overlay_heatmap(base, cam)
        pred_name = CLASS_NAMES[rec["pred"]]

        axes[i].imshow(overlay)
        axes[i].axis("off")
        axes[i].set_title(f"pred={pred_name}\nconf={rec['conf']:.2f}", fontsize=8)

        fig_s, ax_s = plt.subplots(1, 2, figsize=(7, 3.5))
        ax_s[0].imshow(base);    ax_s[0].axis("off"); ax_s[0].set_title("Original")
        ax_s[1].imshow(overlay); ax_s[1].axis("off")
        ax_s[1].set_title(f"Grad-CAM (pred={pred_name}, {rec['conf']:.2f})")
        fig_s.suptitle(f"True=dermatophytosis | sample #{rec['idx']}", fontsize=10)
        fig_s.tight_layout()
        fig_s.savefig(
            out_dir / f"{i:02d}_true-dermatophytosis_pred-{pred_name}_conf{rec['conf']:.2f}.jpg",
            dpi=120, bbox_inches="tight",
        )
        plt.close(fig_s)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Grad-CAM: Top-{n} confident misclassifications of dermatophytosis "
        f"(recall={recall:.1f}%)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "_overview_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ 已輸出 {n} 張個別對照圖 + 總覽 grid 至 {out_dir}")


if __name__ == "__main__":
    main()

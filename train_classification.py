import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from config_classification import config
except ImportError:
    print("找不到 config_classification.py，請先跑 merge_datasets.py")
    exit(1)

CLASS_NAMES = config.CLASS_NAMES

# MPS = Mac M-series GPU, cuda = NVIDIA, fallback to cpu
def get_device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------- Dataset ----------

class DogSkinClassificationDataset(Dataset):

    def __init__(self, data_dir, mode="train"):
        self.data_dir = Path(data_dir)
        self.mode = mode

        with open(self.data_dir / f"{mode}_annotations.json", "r") as f:
            self.annotations = json.load(f)

        self.transform = self._build_transforms()
        print(f"✓ {mode}: {len(self.annotations)} samples loaded")

    def _build_transforms(self):
        if self.mode == "train":
            aug_list = []
            # RandomResizedCrop 取代固定 Resize，覆蓋 config.RANDOM_CROP
            if config.RANDOM_CROP:
                aug_list.append(
                    transforms.RandomResizedCrop(
                        config.IMG_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15)
                    )
                )
            else:
                aug_list.append(transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)))

            aug_list += [
                transforms.RandomHorizontalFlip(p=config.RANDOM_FLIP),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(config.RANDOM_ROTATION),
            ]
            if config.COLOR_JITTER:
                aug_list.append(
                    transforms.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                    )
                )
            aug_list += [
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
            return transforms.Compose(aug_list)
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image = Image.open(self.data_dir / ann["image_path"]).convert("RGB")
        return self.transform(image), ann["label"]


# ---------- Model ----------

def create_model(num_classes, backbone="efficientnet_b0", pretrained=True):
    from torchvision.models import (
        efficientnet_b0, EfficientNet_B0_Weights,
        resnet50, ResNet50_Weights,
        mobilenet_v3_small, MobileNet_V3_Small_Weights,
    )
    if backbone == "resnet50":
        w = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=w)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        w = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=w)
        in_features = model.classifier[1].in_features
        # Dropout 加大至 0.4，抑制過擬合
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    elif backbone == "mobilenet_v3_small":
        w = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=w)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


def set_backbone_trainable(model, backbone, trainable: bool):
    # 凍結/解凍特徵萃取層，分類頭永遠保持可訓練；架構完全不變，只是暫時不更新梯度
    if backbone == "resnet50":
        modules = [m for name, m in model.named_children() if name != "fc"]
    elif backbone in ("efficientnet_b0", "mobilenet_v3_small"):
        modules = [model.features]
    else:
        return
    for m in modules:
        for p in m.parameters():
            p.requires_grad = trainable


# ---------- Loss ----------

class FocalLoss(nn.Module):
    # CrossEntropy 加上 (1-pt)^gamma 的難例權重，讓模型更關注 dermatophytosis
    # 這類容易誤判的類別；label_smoothing 與 class weight 沿用同一套 CE 實作
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(
            weight=weight, label_smoothing=label_smoothing, reduction="none"
        )

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss.detach())
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ---------- MixUp / CutMix ----------

def mixup_cutmix_data(images, labels, alpha_mixup, alpha_cutmix, prob):
    if (alpha_mixup <= 0 and alpha_cutmix <= 0) or np.random.rand() > prob:
        return images, labels, labels, 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    labels_b = labels[index]

    if alpha_cutmix > 0 and np.random.rand() < 0.5:
        lam = float(np.random.beta(alpha_cutmix, alpha_cutmix))
        H, W = images.shape[2], images.shape[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
        cx, cy = np.random.randint(W), np.random.randint(H)
        x1, x2 = int(np.clip(cx - cut_w // 2, 0, W)), int(np.clip(cx + cut_w // 2, 0, W))
        y1, y2 = int(np.clip(cy - cut_h // 2, 0, H)), int(np.clip(cy + cut_h // 2, 0, H))
        images[:, :, y1:y2, x1:x2] = images[index][:, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    elif alpha_mixup > 0:
        lam = float(np.random.beta(alpha_mixup, alpha_mixup))
        images = lam * images + (1 - lam) * images[index]
    else:
        return images, labels, labels, 1.0

    return images, labels, labels_b, lam


# ---------- Chart helpers ----------

def _save(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path.name}")


def plot_loss_accuracy(history, out_dir):
    epochs = range(len(history["train_loss"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_acc"], label="Train", linewidth=2)
    ax.plot(epochs, history["val_acc"],   label="Val",   linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Curve"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, out_dir / "accuracy_curve.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], label="Train", linewidth=2)
    ax.plot(epochs, history["val_loss"],   label="Val",   linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss Curve"); ax.legend(); ax.grid(alpha=0.3)
    _save(fig, out_dir / "loss_curve.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   linewidth=2)
    axes[0].set_title("Loss Curve"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, history["train_acc"], label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_acc"],   label="Val",   linewidth=2)
    axes[1].set_title("Accuracy Curve"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.suptitle("Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir / "training_history.png")


def _plot_single_confusion_matrix(data, title, fmt, class_names, out_path):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(7, n), max(6, n)))
    im = ax.imshow(data, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    thresh = data.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                    fontsize=7, color="white" if data[i, j] > thresh else "black")
    plt.tight_layout()
    _save(fig, out_path)


def plot_confusion_matrix(y_true, y_pred, class_names, out_dir):
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    # 拆成兩個獨立檔案，對齊 YOLO 的 confusion_matrix.png /
    # confusion_matrix_normalized.png 命名慣例
    _plot_single_confusion_matrix(
        cm, "Confusion Matrix (Raw Counts)", "d", class_names,
        out_dir / "confusion_matrix.png",
    )
    _plot_single_confusion_matrix(
        cm_norm, "Confusion Matrix (Normalized)", ".2f", class_names,
        out_dir / "confusion_matrix_normalized.png",
    )


def plot_roc_curves(y_true, y_prob, class_names, out_dir):
    n = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n)))

    fpr_d, tpr_d, auc_d = {}, {}, {}
    for i in range(n):
        fpr_d[i], tpr_d[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc_d[i] = auc(fpr_d[i], tpr_d[i])

    all_fpr  = np.unique(np.concatenate([fpr_d[i] for i in range(n)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n):
        mean_tpr += np.interp(all_fpr, fpr_d[i], tpr_d[i])
    mean_tpr /= n
    macro_auc = auc(all_fpr, mean_tpr)

    colors = plt.cm.tab10(np.linspace(0, 1, n))
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (name, color) in enumerate(zip(class_names, colors)):
        ax.plot(fpr_d[i], tpr_d[i], color=color, linewidth=1.5,
                label=f"{name} (AUC={auc_d[i]:.3f})")
    ax.plot(all_fpr, mean_tpr, "k--", linewidth=2,
            label=f"Macro Avg (AUC={macro_auc:.3f})")
    ax.plot([0, 1], [0, 1], "gray", linewidth=1, linestyle=":")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=7); ax.grid(alpha=0.3)
    _save(fig, out_dir / "roc_curves.png")


def plot_pr_curves(y_true, y_prob, class_names, out_dir):
    n = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n)))

    colors = plt.cm.tab10(np.linspace(0, 1, n))
    fig, ax = plt.subplots(figsize=(9, 7))
    ap_list = []
    for i, (name, color) in enumerate(zip(class_names, colors)):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        ap_list.append(ap)
        ax.plot(recall, precision, color=color, linewidth=1.5,
                label=f"{name} (AP={ap:.3f})")
    macro_ap = np.mean(ap_list)
    ax.axhline(y=macro_ap, color="k", linestyle="--", linewidth=1.5,
               label=f"Macro Avg AP={macro_ap:.3f}")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (AUC-PR)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=7); ax.grid(alpha=0.3)
    _save(fig, out_dir / "pr_curves.png")


def plot_confidence_curves(y_true, y_prob, class_names, out_dir):
    # P/R/F1 vs Confidence threshold，對齊 YOLO 的
    # P_curve.png / R_curve.png / F1_curve.png
    n = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n)))
    x = np.linspace(0, 1, 1000)
    colors = plt.cm.tab10(np.linspace(0, 1, n))

    P  = np.zeros((n, len(x)))
    R  = np.zeros((n, len(x)))
    F1 = np.zeros((n, len(x)))

    for i in range(n):
        precision, recall, thresholds = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        p, r = precision[:-1], recall[:-1]
        f1 = 2 * p * r / (p + r + 1e-16)

        order = np.argsort(thresholds)
        th_sorted = thresholds[order]
        P[i]  = np.interp(x, th_sorted, p[order],  left=p[order][0],  right=p[order][-1])
        R[i]  = np.interp(x, th_sorted, r[order],  left=r[order][0],  right=r[order][-1])
        F1[i] = np.interp(x, th_sorted, f1[order], left=f1[order][0], right=f1[order][-1])

    for data, ylabel, fname in [(P, "Precision", "P_curve.png"),
                                 (R, "Recall", "R_curve.png"),
                                 (F1, "F1", "F1_curve.png")]:
        mean_curve = data.mean(axis=0)
        best_x, best_y = x[np.argmax(mean_curve)], mean_curve.max()

        fig, ax = plt.subplots(figsize=(9, 6))
        for i, (name, color) in enumerate(zip(class_names, colors)):
            ax.plot(x, data[i], color=color, linewidth=1, label=name)
        ax.plot(x, mean_curve, "-", color="blue", linewidth=3,
                label=f"all classes {best_y:.2f} at {best_x:.3f}")
        ax.set_xlabel("Confidence"); ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title(f"{ylabel}-Confidence Curve", fontsize=13, fontweight="bold")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        _save(fig, out_dir / fname)


def plot_label_distribution(config, class_names, out_dir):
    # 各類別在 train/val 的樣本數，對齊 YOLO 的 labels.jpg
    with open(config.TRAIN_DIR / "train_annotations.json", "r") as f:
        train_anns = json.load(f)
    with open(config.VAL_DIR / "val_annotations.json", "r") as f:
        val_anns = json.load(f)

    n = len(class_names)
    train_counts = np.zeros(n, dtype=int)
    val_counts   = np.zeros(n, dtype=int)
    for a in train_anns:
        train_counts[a["label"]] += 1
    for a in val_anns:
        val_counts[a["label"]] += 1

    x = np.arange(n); w = 0.35
    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 6))
    bars1 = ax.bar(x - w / 2, train_counts, w, label="Train", color="#4C72B0")
    bars2 = ax.bar(x + w / 2, val_counts,   w, label="Val",   color="#DD8452")
    ax.bar_label(bars1, fontsize=7, padding=2)
    ax.bar_label(bars2, fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Instance Count")
    ax.set_title("Dataset Class Distribution", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir / "labels.png")


def plot_feature_importance(model, class_names, out_dir, backbone):
    # FC 層 weight 的 L2 norm，反映模型對各類別的關注強度
    # 如果需要看病灶位置，要用 Grad-CAM，這裡只是權重強度
    try:
        if backbone == "resnet50":
            w = model.fc.weight.detach().cpu().numpy()
        elif backbone == "efficientnet_b0":
            w = model.classifier[1].weight.detach().cpu().numpy()
        elif backbone == "mobilenet_v3_small":
            w = model.classifier[3].weight.detach().cpu().numpy()
        else:
            print("  ⚠ feature importance: unsupported backbone, skipped")
            return
    except Exception as e:
        print(f"  ⚠ feature importance failed: {e}")
        return

    norms      = np.linalg.norm(w, axis=1)
    idx_sorted = np.argsort(norms)[::-1]

    fig, ax = plt.subplots(figsize=(9, max(5, len(class_names) * 0.6)))
    bars = ax.barh(
        [class_names[i] for i in idx_sorted],
        norms[idx_sorted],
        color=plt.cm.viridis(np.linspace(0.2, 0.8, len(class_names))),
    )
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xlabel("L2 Norm of FC Weight Vector")
    ax.set_title(
        "Classifier Attention Strength per Class (FC Weight L2 Norm)\n"
        "Higher = stronger feature response  |  For lesion localization use Grad-CAM",
        fontsize=10, fontweight="bold",
    )
    ax.invert_yaxis(); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir / "feature_importance.png")


def plot_per_class_metrics(y_true, y_pred, class_names, out_dir):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    n = len(class_names)
    x = np.arange(n)
    w = 0.26

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 6))
    ax.bar(x - w, precision, w, label="Precision", color="#4C72B0")
    ax.bar(x,     recall,    w, label="Recall",    color="#DD8452")
    ax.bar(x + w, f1,        w, label="F1-Score",  color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1-Score", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, out_dir / "per_class_metrics.png")


def generate_all_charts(trainer, y_true, y_pred, y_prob, class_names, out_dir):
    print("\n[Charts] Generating evaluation charts...")
    plot_loss_accuracy(trainer.history, out_dir)
    plot_confusion_matrix(y_true, y_pred, class_names, out_dir)
    plot_roc_curves(y_true, y_prob, class_names, out_dir)
    plot_pr_curves(y_true, y_prob, class_names, out_dir)
    plot_confidence_curves(y_true, y_prob, class_names, out_dir)
    plot_label_distribution(trainer.config, class_names, out_dir)
    plot_feature_importance(trainer.model, class_names, out_dir, config.BACKBONE)
    plot_per_class_metrics(y_true, y_pred, class_names, out_dir)
    print(f"[Charts] Done. Saved to: {out_dir}")


# ---------- Trainer ----------

class Trainer:

    def __init__(self, config):
        self.config = config
        self.device = get_device()   # MPS / CUDA / CPU 自動偵測

        print(f"\nDevice : {self.device}")
        print(f"Backbone: {config.BACKBONE}")

        self.model = create_model(
            config.NUM_CLASSES, config.BACKBONE, config.PRETRAINED
        ).to(self.device)

        if config.FREEZE_BACKBONE_EPOCHS > 0:
            set_backbone_trainable(self.model, config.BACKBONE, trainable=False)
            print(f"✓ Backbone frozen for first {config.FREEZE_BACKBONE_EPOCHS} epochs "
                  f"(only classifier head trains)")

        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Params  : {total:,}  (trainable: {trainable:,})")

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        # Warmup 階段：從 LR/10 線性增加到 LR，之後接 CosineAnnealing
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.WARMUP_EPOCHS,
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, config.NUM_EPOCHS - config.WARMUP_EPOCHS),
            eta_min=config.MIN_LR,
        )
        self.scheduler = optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.WARMUP_EPOCHS],
        )
        class_weights = self._class_weights()
        if config.USE_FOCAL_LOSS:
            self.criterion = FocalLoss(
                weight=class_weights, gamma=config.FOCAL_GAMMA,
                label_smoothing=config.LABEL_SMOOTHING,
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=config.LABEL_SMOOTHING,
            )

        self.current_epoch = 0
        self.best_acc      = 0.0
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [], "val_f1": [],
        }

    def _class_weights(self):
        # 用 inverse frequency 讓稀少類別有更高的 loss 權重
        with open(self.config.TRAIN_DIR / "train_annotations.json", "r") as f:
            anns = json.load(f)

        counts = np.zeros(self.config.NUM_CLASSES, dtype=np.float32)
        for a in anns:
            counts[a["label"]] += 1

        counts  = np.where(counts == 0, 1, counts)
        weights = counts.sum() / (self.config.NUM_CLASSES * counts)
        tensor  = torch.tensor(weights, dtype=torch.float32).to(self.device)

        print("\nClass weights:")
        for i, (name, w, n) in enumerate(zip(CLASS_NAMES, weights, counts)):
            print(f"  [{i}] {name:35s}  w={w:.4f}  n={int(n)}")

        return tensor

    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        cfg = self.config

        pbar = tqdm(loader, desc=f"Epoch {self.current_epoch}")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            images, labels_a, labels_b, lam = mixup_cutmix_data(
                images, labels, cfg.MIXUP_ALPHA, cfg.CUTMIX_ALPHA, cfg.MIXUP_PROB
            )

            out  = self.model(images)
            loss = lam * self.criterion(out, labels_a) + (1 - lam) * self.criterion(out, labels_b)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, pred = out.max(1)
            total   += labels.size(0)
            # MixUp/CutMix 期間用兩個標籤依 lam 加權估計訓練準確率，僅供參考
            correct += (lam * pred.eq(labels_a).sum().item()
                        + (1 - lam) * pred.eq(labels_b).sum().item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

        return total_loss / len(loader), 100. * correct / total

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(self.device), labels.to(self.device)
            out  = self.model(images)
            loss = self.criterion(out, labels)
            total_loss += loss.item()

            probs = torch.softmax(out, dim=1)
            _, pred = out.max(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        val_loss = total_loss / len(loader)
        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        return (
            {"val_loss": val_loss, "accuracy": acc,
             "precision": prec, "recall": rec, "f1_score": f1},
            all_preds, all_labels, np.array(all_probs),
        )

    @torch.no_grad()
    def evaluate_tta(self, loader):
        # 只用在最終評估：原圖 + 水平翻轉各跑一次，平均 softmax 機率再決定預測類別
        # 訓練期間的逐 epoch validate() 不套用，避免拖慢訓練速度
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []

        for images, labels in tqdm(loader, desc="Final Eval (TTA)"):
            images, labels = images.to(self.device), labels.to(self.device)

            probs      = torch.softmax(self.model(images), dim=1)
            probs_flip = torch.softmax(self.model(torch.flip(images, dims=[3])), dim=1)
            probs_avg  = (probs + probs_flip) / 2

            pred = probs_avg.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs_avg.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        return (
            {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1},
            all_preds, all_labels, np.array(all_probs),
        )

    def save_checkpoint(self, metrics, is_best=False):
        ckpt = {
            "epoch":                self.current_epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_acc":             self.best_acc,
            "metrics":              metrics,
            "history":              self.history,
        }
        # 永遠覆寫 latest
        torch.save(ckpt, self.config.WEIGHTS_DIR / "latest.pth")

        # 每隔幾個 epoch 存一份帶編號的備份，防止 latest 損壞
        if self.current_epoch % self.config.SAVE_CHECKPOINT_FREQ == 0:
            torch.save(ckpt, self.config.WEIGHTS_DIR / f"model_ep{self.current_epoch:04d}.pth")

        if is_best:
            torch.save(ckpt, self.config.WEIGHTS_DIR / "best_model.pth")
            print(f"    ✓ New best model!  Acc: {metrics['accuracy']:.4f}")

    def train(self, train_loader, val_loader, test_loader=None):
        print("\n" + "=" * 60)
        print("Training Start")
        print("=" * 60)

        patience = 0
        best_f1  = 0.0
        best_val_loss = float("inf")

        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch = epoch

            if (self.config.FREEZE_BACKBONE_EPOCHS > 0
                    and epoch == self.config.FREEZE_BACKBONE_EPOCHS):
                set_backbone_trainable(self.model, self.config.BACKBONE, trainable=True)
                print(f"✓ Backbone unfrozen at epoch {epoch}, fine-tuning full model")

            tr_loss, tr_acc = self.train_epoch(train_loader)
            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)

            val_metrics, _, _, _ = self.validate(val_loader)
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_acc"].append(val_metrics["accuracy"] * 100)
            self.history["val_f1"].append(val_metrics["f1_score"])

            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"\nEpoch {epoch}  lr={current_lr:.2e}:")
            print(f"  Train  Loss={tr_loss:.4f}  Acc={tr_acc:.2f}%")
            print(f"  Val    Loss={val_metrics['val_loss']:.4f}  Acc={val_metrics['accuracy']:.4f}"
                  f"  P={val_metrics['precision']:.4f}  R={val_metrics['recall']:.4f}"
                  f"  F1={val_metrics['f1_score']:.4f}")

            # best model 仍以 weighted F1 為基準，比 accuracy 在類別不平衡情況下更穩定
            is_best = val_metrics["f1_score"] > best_f1
            if is_best:
                best_f1 = val_metrics["f1_score"]
                self.best_acc = val_metrics["accuracy"]

            # early stopping 改監控 val_loss 是否持續惡化：這份資料 val_loss 在
            # epoch ~5 後就已經沒有下降趨勢、甚至劇烈震盪，用 val_loss 比等 F1
            # 停滯更早抓到過擬合，兩者判準拆開、互不影響 best checkpoint 的選擇
            if val_metrics["val_loss"] < best_val_loss - self.config.EARLY_STOPPING_MIN_DELTA:
                best_val_loss = val_metrics["val_loss"]
                patience = 0
            else:
                patience += 1

            if epoch % 5 == 0 or is_best:
                self.save_checkpoint(val_metrics, is_best)

            if patience >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}  "
                      f"(val_loss plateaued, best F1={best_f1:.4f})")
                break

        print("\n" + "=" * 60)
        print(f"Training complete.  Best val F1={best_f1:.4f}  Acc={self.best_acc:.4f}")
        print("=" * 60)

        # 用 test set（held-out，從未參與訓練或 early stopping）做最終評估，
        # 避免 val set 同時被拿來挑 checkpoint 又拿來當成效報告，導致指標偏樂觀
        best_ckpt_path = self.config.WEIGHTS_DIR / "best_model.pth"
        if best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"✓ Loaded best model (epoch {ckpt['epoch']}) for final evaluation")

        eval_loader, eval_name = (test_loader, "TEST") if test_loader is not None else (val_loader, "VAL")
        if test_loader is None:
            print("⚠ test set not found, falling back to val set for final evaluation")

        if self.config.TTA_ENABLED:
            eval_name += "+TTA"
            eval_metrics, eval_preds, eval_labels, eval_probs = self.evaluate_tta(eval_loader)
        else:
            eval_metrics, eval_preds, eval_labels, eval_probs = self.validate(eval_loader)
        print(f"\n[{eval_name}] Acc={eval_metrics['accuracy']:.4f}  "
              f"P={eval_metrics['precision']:.4f}  R={eval_metrics['recall']:.4f}  "
              f"F1={eval_metrics['f1_score']:.4f}")

        generate_all_charts(
            self, eval_labels, eval_preds, eval_probs,
            CLASS_NAMES, self.config.RESULTS_DIR,
        )


# ---------- Main ----------

def main():
    print("=" * 60)
    print("Dog Skin Disease Classification")
    print("=" * 60)

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    config.create_directories()

    if not (config.TRAIN_DIR / "train_annotations.json").exists():
        print("Training data not found. Run merge_datasets.py first.")
        return

    print("\nLoading data...")
    train_ds = DogSkinClassificationDataset(config.TRAIN_DIR, mode="train")
    val_ds   = DogSkinClassificationDataset(config.VAL_DIR,   mode="val")

    test_ds = None
    if (config.TEST_DIR / "test_annotations.json").exists():
        test_ds = DogSkinClassificationDataset(config.TEST_DIR, mode="test")

    device     = get_device()
    pin_memory = device.type == "cuda"

    # WeightedRandomSampler：讓每個 batch 中各類別出現機率接近均等
    # 解決 healthy 樣本數遠多於其他類的不平衡問題
    from collections import Counter
    from torch.utils.data import WeightedRandomSampler
    label_counts = Counter(ann["label"] for ann in train_ds.annotations)
    sample_weights = [
        1.0 / label_counts[ann["label"]] for ann in train_ds.annotations
    ]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                              sampler=sampler,        # shuffle 與 sampler 不能共存
                              num_workers=config.NUM_WORKERS,
                              pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=config.NUM_WORKERS,
                              pin_memory=pin_memory)

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                                 shuffle=False, num_workers=config.NUM_WORKERS,
                                 pin_memory=pin_memory)

    Trainer(config).train(train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()
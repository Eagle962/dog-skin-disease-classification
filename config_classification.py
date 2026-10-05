import torch
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

    NUM_CLASSES = 7
    CLASS_NAMES = ['bacterial_dermatosis', 'dermatitis', 'dermatophytosis', 'flea_allergy_dermatitis', 'healthy', 'hypersensitivity_dermatitis', 'mange']
    CLASS_TO_IDX = {'bacterial_dermatosis': 0, 'dermatitis': 1, 'dermatophytosis': 2, 'flea_allergy_dermatitis': 3, 'healthy': 4, 'hypersensitivity_dermatitis': 5, 'mange': 6}

    # efficientnet_b0 在中小型資料集通常比 resnet50 穩
    BACKBONE   = "efficientnet_b0"
    PRETRAINED = True
    IMG_SIZE   = 224

    BATCH_SIZE    = 32
    NUM_EPOCHS    = 60
    LEARNING_RATE = 1e-3
    # 過擬合明顯（train acc 96% vs val acc ~72%），weight_decay 從 1e-3 提高到 2e-3
    WEIGHT_DECAY  = 2e-3

    RANDOM_FLIP     = 0.5
    RANDOM_ROTATION = 15
    COLOR_JITTER    = True
    RANDOM_CROP     = True

    LR_SCHEDULER   = "cosine"
    WARMUP_EPOCHS  = 3
    MIN_LR         = 1e-6

    # ---- 抗過擬合 / 損失函數強化（不改動模型架構本身） ----
    LABEL_SMOOTHING = 0.1

    # dermatophytosis 等易混淆類別 recall 很低，Focal Loss 會加重難分樣本的梯度
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA    = 2.0

    # 每個 batch 有 MIXUP_PROB 的機率套用 MixUp 或 CutMix 其中之一
    MIXUP_ALPHA  = 0.2
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB   = 0.5

    # 前 N 個 epoch 凍結 backbone、只訓練分類頭，之後解凍做全模型微調
    # 小資料集直接全模型訓練很容易在前幾個 epoch 就開始過擬合
    FREEZE_BACKBONE_EPOCHS = 5

    # 最終評估時用水平翻轉做 Test-Time Augmentation，平均機率再取預測類別
    TTA_ENABLED = True

    EVAL_METRICS = ["accuracy", "precision", "recall", "f1_score"]

    SAVE_CHECKPOINT_FREQ    = 5
    KEEP_BEST_N             = 3
    # early stopping 改監控 val_loss 是否持續惡化（val_loss 在 epoch ~5 後就已經
    # 沒有下降趨勢，比等 weighted F1 停滯更早抓到過擬合），patience 從 15 收緊到 8
    EARLY_STOPPING_PATIENCE  = 8
    EARLY_STOPPING_MIN_DELTA = 1e-3

    TENSORBOARD_LOG = False

    @classmethod
    def create_directories(cls):
        for d in [cls.DATA_DIR, cls.OUTPUT_DIR, cls.WEIGHTS_DIR,
                  cls.LOGS_DIR, cls.RESULTS_DIR]:
            d.mkdir(parents=True, exist_ok=True)


config = Config()
config.create_directories()
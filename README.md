# 汪星人 AI 皮膚醫生 — 執行方式

全部請在專案資料夾內，用專案的 venv 執行（`venv/bin/python`，或先 `source venv/bin/activate`）。
缺套件時：`venv/bin/pip install -r requirements.txt`

| 步驟 | 指令 | 說明 |
|---|---|---|
| 1. 下載來源（已下載則跳過） | `python download_sources.py` | 需 Roboflow API key（環境變數 `ROBOFLOW_API_KEY`，或 `~/.config/woofai-creds/roboflow.json`） |
| 2. 建立資料集 | `python build_dataset.py` | 約 80 秒；輸出 `dataset_cc_by/`（會覆寫） |
| 3. 檢查資料集與設定 | `python patch_annotations.py` | 類別順序、檔案是否存在 |
| 4. 影像品質稽核（選用） | `python audit_image_quality.py` | 標出旋轉黑邊、過小裁切；輸出到 `outputs_cc_by/results/` |
| 5. 訓練與評估 | `python train_classification.py` | 最多 60 epoch，early stopping patience 15（`WOOF_PATIENCE`）；輸出 `outputs_cc_by/`（權重、圖表）。這台機器 1 epoch 約 1.7 分鐘 |
| 6. 推論 + Grad-CAM | `python predict.py 圖片或資料夾` | 需先完成訓練 |
| 7. 錯誤分析（選用） | `python gradcam_dermatophytosis.py` | dermatophytosis 誤判對照圖 |

冒煙測試（只訓練 1 epoch、輸出到別處）：
`WOOF_EPOCHS=1 WOOF_OUTPUT_DIR=/tmp/smoke python train_classification.py`

資料來源與授權：`dataset_sources.md`。舊版腳本在 `legacy/`（勿執行）。
舊的 `outputs/ outputs2.0/ outputs3.0/ old_output/`（在含未授權來源的舊資料上訓練）已於 2026-10-12 刪除，其準確率（92.5%、96.1%）已作廢，不可引用。目前只保留 `outputs_cc_by/`（B0，主模型）與 `outputs_b2/`（B2 對照實驗）。

## 結果與部署
- 主模型 EfficientNet-B0 @224（預設設定）：新切分測試 Accuracy **77.72%**（286/368，TTA）。曾用舊切分得到 80.16%，但稽核發現舊測試集有 16.8% 在訓練集有鏡像／小角度旋轉副本，已改版分組並重訓（詳見 `docs/修改說明.md`）。B2 @256 + 強增強是在舊切分上的對照實驗（74.73%，未在新切分重做）；設定見 `config_classification.py` 的環境變數（`WOOF_BACKBONE`、`WOOF_IMG_SIZE`、`WOOF_STRONG_AUG`、`WOOF_OUTPUT_DIR`）。
- App 後端 `main.py`（FastAPI）：`MODEL_PATH`、`WOOF_BACKBONE`、`WOOF_IMG_SIZE` 設定；需額外安裝 `fastapi python-multipart opencv-python-headless`（見 `requirements.txt`）。
- `verify_deploy_parity.py`：用 main.py 的推論路徑預測測試集並輸出 CSV，用於比對不同環境（例如 Docker torch 2.3）。
- `gen_doc.py`：由 `開源平台佐證文件.md` 產生 docx / pdf（需 python-docx、reportlab）。

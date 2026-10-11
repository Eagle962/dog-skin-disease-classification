# 資料來源與授權（dataset_cc_by/ 實際使用）

授權由 Roboflow API 的專案 metadata 與匯出 README 取得（2026-10-10），非僅依網頁搜尋摘要。

| 來源 | 作者／Workspace | 專案 | 版本 | 授權 | 本機資料夾 | 樣本數（去重、剔除黑邊後） |
|---|---|---|---|---|---|---|
| litespy | litespy-l22hu | dog-skin-diseases | v1 | CC BY 4.0 | Dog-Skin-Diseases-1 | 1,274 |
| smartdog | smartdog-f8su3 | dog-skin-disease-dataset-bzhug | v1 | CC BY 4.0 | Dog-Skin-Disease-Dataset-1 | 1,180 |
| gabungan | research-on-health | gabungan-of798 | v1 | CC BY 4.0 | gabungan-1 | 2,004 |
| smoky | Smoky（smoky） | dogs-skin-disease-manual | v2 | CC BY 4.0（API：project.license） | candidates/rf_smoky | 278 |

改作：類別重新對映、依標註框裁切（padding 15%）、去重、重新切分。Smoky 的 `tumor` 類別（不在 7 類內）已丟棄。
引用時需標示作者、授權連結（https://creativecommons.org/licenses/by/4.0/）與「已修改」。

## 已評估但不採用
| 來源 | 原因 |
|---|---|
| Skin Disease of Dog（maria-angelica-krgdu） | 無授權、metadata 為 Private；已自本機刪除 |
| MediPaw-SkinDerm6（IEEE Dataport） | 授權未載明、需付費訂閱、來源不明 |
| Roboflow pethealth/dog-skin-diseases-bf171（CC BY 4.0） | 所有版本皆含前處理或擴增；下載的 v3 與 litespy 同為近單色色偏圖（檔名 picN_jpg 相同），資訊量低且近重複無法用 pHash 可靠判定，疑為 litespy 分支 |
| Kaggle yashmotiani/dogs-skin-disease-dataset（CC0，上傳者聲明） | 442 張中 65% 與現有資料重複（疑為 smartdog 來源）；其餘混有色偏圖與帶黑邊的旋轉擴增圖 |
| Kaggle youssefmohmmed/dogs-skin-diseases-image-dataset（Apache 2.0，上傳者聲明） | 檔名含 `_x16_jpg.rf.`，為 Roboflow 擴增再上傳，來源不明；未下載 |

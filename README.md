# TextVP - Unofficial Implementation

> **Textualize Visual Prompt for Image Editing via Diffusion Bridge** (AAAI'25)  
> ⚠️ 這是非官方的實作版本

## 簡介

本專案是論文 "Textualize Visual Prompt for Image Editing via Diffusion Bridge" 的非官方實現，基於 [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt) 框架開發。

![DEMO](./experiments/20251203_014400/cross_replace=[0.2,%201.0],self_replace=0.0,encoded=False,guidance_scale=3.5,/test_image1.png)

## 專案結構

```
├── experiment_config.py     # 實驗配置管理
├── ptp_utils.py            # Prompt-to-Prompt 工具函數
├── seq_aligner.py          # 序列對齊工具
├── image_utils.py          # 圖像處理工具
├── inversion.py            # DDIM Inversion
├── main.ipynb              # 主要訓練/測試 Notebook
├── data_generator/         # 資料生成工具
│   └── prompt2prompt_gen_datapair.ipynb # 生成 Prompt-to-Prompt 資料對 (prompt-to-prompt 版本)
│   └── inp2p.py            # 生成 Prompt-to-Prompt 資料對(intructpix2pix 版本)
├── dataset/                # 資料集
└── experiments/            # 實驗輸出目錄
```

## 安裝

```bash
pip install -r requirements.txt
```

## 使用方法

1. 準備資料集放置於 `dataset/` 目錄
2. 使用 `main.ipynb` 進行訓練和測試
3. 實驗結果會自動保存至 `experiments/` 目錄

## 推論 (Sampling)

使用 `sample.py` 對圖片進行風格轉換：

```bash
python sample.py \
    --checkpoint path/to/epoch_X.pt \
    --config path/to/train_config.json \
    --image_dir path/to/input/images \
    --output_dir path/to/output/images
```

### 範例

```bash
python sample.py \
    -c ./experiments/20251203_014400/cross_replace=\[0.2,\ 1.0\],self_replace=0.0,encoded=False,guidance_scale=3.5,/epoch_13.pt \
    -cfg ./experiments/20251203_014400/cross_replace=\[0.2,\ 1.0\],self_replace=0.0,encoded=False,guidance_scale=3.5,/train_config.json \
    -i ./dataset_old/test_1130(single_data) \
    -o ./sample
```

### 參數說明

| 參數 | 縮寫 | 說明 |
|------|------|------|
| `--checkpoint` | `-c` | 訓練好的 `.pt` 檔案路徑 |
| `--config` | `-cfg` | 對應的 `train_config.json` 路徑 |
| `--image_dir` | `-i` | 輸入圖片目錄（或 glob pattern） |
| `--output_dir` | `-o` | 輸出目錄 |
| `--device` | `-d` | 使用的設備（預設 `cuda:0`） |
| `--ext` | `-e` | 圖片副檔名（預設 `png`） |

## 致謝

- 基於 [google/prompt-to-prompt](https://github.com/google/prompt-to-prompt) 開發
- 原論文: [Textualize Visual Prompt for Image Editing via Diffusion Bridge (AAAI'25)](https://arxiv.org/abs/2501.03495)

## Disclaimer

This is an unofficial implementation and is not affiliated with the original authors.

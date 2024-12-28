# 車牌偵測與字符識別系統

## 專案概述
這個專案實現了一個自動化的車牌偵測和字符識別系統，使用 YOLOv8 進行車牌定位，並結合 Tesseract OCR 進行字符識別。

## 功能特點
- 使用 YOLOv8 進行車牌區域偵測
- 應用 Tesseract OCR 進行字符識別
- 支援批量處理圖片
- 輸出處理結果到多個資料夾
- 生成標準化的標記文件

## 環境需求
- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- Tesseract OCR
- NumPy

## 安裝步驟
1. 安裝 Python 依賴：
```bash
pip install ultralytics opencv-python pytesseract numpy
```

2. 安裝 Tesseract OCR：
- macOS：
```bash
brew install tesseract
```
- Windows：
從 [GitHub Releases](https://github.com/UB-Mannheim/tesseract/wiki) 下載安裝

3. 下載預訓練模型：
- 使用來自 [Kaggle](https://www.kaggle.com/models/pythonistasamurai/yolov8-license-plate-detection) 的預訓練模型

## 目錄結構
```
├── main.py              # 主程式
├── best.pt             # YOLOv8 預訓練模型
├── sample 2024/        # 待處理圖片目錄
├── result/             # YOLO 偵測結果
├── labels/             # YOLO 標籤檔案
├── number/            # 字符識別結果
└── 411185011.txt      # 最終輸出結果
```

## 使用方法
1. 將待處理圖片放入 `sample 2024` 資料夾
2. 執行主程式：
```bash
python main.py
```

## 輸出說明
- `result/`: 存放 YOLO 模型的車牌偵測結果
- `labels/`: 包含 YOLO 格式的標籤文件
- `number/`: 存放字符識別後的結果圖片
- `411185011.txt`: 包含所有處理結果的文本文件

## 模型來源
- YOLOv8 車牌偵測模型來自 [Kaggle](https://www.kaggle.com/models/pythonistasamurai/yolov8-license-plate-detection)

## 注意事項
- 確保 Tesseract 路徑正確設定
- 建議使用高解析度且清晰的圖片以獲得最佳效果
- 處理結果的準確度可能受圖片品質影響
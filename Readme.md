# 車牌偵測與字符識別系統

## 專案概述
這個專案實現了一個自動化的車牌偵測和字符識別系統，使用 YOLOv8 進行車牌定位，並結合 Tesseract OCR 進行字符識別。系統具備智能邊緣檢測功能，即使在車牌邊緣不完整的情況下也能進行處理。

## 功能特點
- 使用 YOLOv8 進行車牌區域偵測，並自動選擇信心分數最高的車牌
- 智能車牌邊緣處理：
  - 優先使用完整車牌邊緣
  - 當無法檢測到完整邊緣時，自動使用圖片邊界作為參考
  - 支援部分邊緣缺失的車牌處理
- 應用 Tesseract OCR 進行字符識別
- 支援批量處理圖片
- 輸出多層次的處理結果

## 處理流程
1. 車牌偵測 (YOLO)：選擇信心分數最高的車牌區域
2. 邊緣處理：
   - 嘗試檢測車牌完整邊緣
   - 若檢測失敗，使用圖片邊界作為替代
3. 透視轉換：將傾斜的車牌轉換為正視圖
4. 字符識別：使用 Tesseract 進行 OCR
5. 結果輸出：生成標記文件和視覺化結果

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
├── perspective.py       # 車牌透視轉換和邊緣處理
├── best.pt             # YOLOv8 預訓練模型
├── result/             # YOLO 偵測結果
├── board/              # 車牌區域截圖
├── processed_board/    # 校正後的車牌圖片
├── labels/             # YOLO 標籤檔案
├── number/             # 字符識別結果
└── 411185011.txt      # 最終輸出結果
```

## 使用方法
1. 將待處理圖片放入指定資料夾
2. 執行主程式：
```bash
python main.py
```

## 輸出說明
- `result/`: 存放 YOLO 模型的車牌偵測結果
- `board/`: 原始車牌截圖
- `processed_board/`: 經過透視轉換後的車牌圖片
- `labels/`: 包含 YOLO 格式的標籤文件
- `number/`: 存放字符識別後的結果圖片(使用透視轉換後圖片偵測並且對應回原圖座標)
- `411185011.txt`: 包含所有處理結果的文本文件

## 注意事項
- 確保 Tesseract 路徑正確設定
- 系統能處理不完整邊緣的車牌，但完整清晰的車牌會有更好的識別效果
- 處理結果的準確度可能受以下因素影響：
  - 圖片解析度和清晰度
  - 車牌邊緣的完整性
  - 光線條件和拍攝角度
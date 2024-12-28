from ultralytics import YOLO  # required for loading the model
import os

MODEL = YOLO("best.pt")  # loading the models

# 指定儲存目錄為當前目錄
current_directory = os.getcwd()

# 預測並儲存結果到當前目錄
MODEL.predict("sample 2024/020.jpg", save=True, save_txt=True, project=current_directory, name="prediction_result", show_labels=False)

#存在當前目錄


import os
from ultralytics import YOLO
import pytesseract
import cv2

MODEL = YOLO("best.pt")
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
def process_image(image_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]  # 獲取圖片尺寸
    
    # 確保 result 和 labels 資料夾存在
    if not os.path.exists('result'):
        os.makedirs('result')
    if not os.path.exists('labels'):
        os.makedirs('labels')
    
    # 使用 YOLO 模型進行車牌偵測
    results = MODEL.predict(image_path, save=False)  # 不直接保存
    
    # 手動保存結果到指定位置
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 保存預測圖片
    for r in results:
        im_array = r.plot()  # 繪製預測結果
        cv2.imwrite(f'result/{file_name}.jpg', im_array)
        
        # 保存標籤文件
        if r.boxes:
            with open(f'labels/{file_name}.txt', 'w') as f:
                for box in r.boxes:
                    # 轉換為YOLO格式 (class x_center y_center width height)
                    x_center = box.xywhn[0][0].item()
                    y_center = box.xywhn[0][1].item()
                    width = box.xywhn[0][2].item()
                    height = box.xywhn[0][3].item()
                    class_id = box.cls[0].item()
                    f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
    # 讀取對應的 label 文件
    file_name = image_path.split('/')[-1].split('.')[0]
    label_path = f"labels/{file_name}.txt"
    
    if not os.path.exists(label_path):
        print(f"未找到標籤文件: {label_path}")
        return
        
    result_lines = [f"{file_name}.jpg"]
    char_bboxes = []

    # 讀取 YOLO 格式的標籤文件
    with open(label_path, 'r') as f:
        for line in f:
            # YOLO 格式: class x_center y_center width height
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            
            # 轉換為像素座標
            x = int((x_center - width/2) * w)
            y = int((y_center - height/2) * h)
            plate_w = int(width * w)
            plate_h = int(height * h)
            
            # 裁剪車牌區域
            plate = img[y:y+plate_h, x:x+plate_w]
            if plate.size == 0:
                continue
                
            plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            _, plate_thresh = cv2.threshold(plate_gray, 127, 255, cv2.THRESH_BINARY)

            # 使用 Tesseract 進行字符識別
            custom_config = r'--oem 3 --psm 8'
            text = pytesseract.image_to_string(plate_thresh, config=custom_config)
            print(f"偵測到字元: {text.strip()}")

            # 獲取字符位置
            boxes = pytesseract.image_to_boxes(plate_thresh, config=custom_config)
            for b in boxes.splitlines():
                b = b.split()
                if len(b) >= 5:
                    x1_char, y1_char, x2_char, y2_char = map(int, b[1:5])
                    char_w = x2_char - x1_char
                    char_h = y2_char - y1_char
                    # 調整為全局座標系統
                    global_x = x + x1_char
                    global_y = y + y1_char
                    char_bboxes.append((global_x, global_y, char_w, char_h))
                    
                    # 在原圖上繪製字符框
                    cv2.rectangle(img, 
                                (global_x, global_y),
                                (global_x + char_w, global_y + char_h),
                                (0, 255, 0), 2)

    # 確保 number 資料夾存在
    if not os.path.exists('number'):
        os.makedirs('number')
    
    # 儲存結果使用原始檔名
    cv2.imwrite(f'number/{file_name}.jpg', img)

    # 寫入結果到 txt 檔
    result_lines.append(str(len(char_bboxes)))
    for bbox in char_bboxes:
        result_lines.append(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")

    with open("411185011.txt", 'a') as f:
        f.write("\n".join(result_lines) + "\n")

def main():
    with open('411185011.txt', 'w') as f:
        f.write("")

    folder_path = 'sample 2024'
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing: {image_file}")
        process_image(image_path)

if __name__ == "__main__":
    main()

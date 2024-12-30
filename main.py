import os
from ultralytics import YOLO
import pytesseract
import cv2
from perspective import process_license_plate
import numpy as np

MODEL = YOLO("best.pt")
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def process_image(image_path):
    # 讀取原始圖片
    img = cv2.imread(image_path)
    if img is None:
        print(f"無法讀取圖片: {image_path}")
        return
    h, w = img.shape[:2]  # 獲取圖片尺寸
    
    # 確保所需資料夾存在
    folders = ['result', 'labels', 'board', 'number']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # 使用 YOLO 模型進行車牌偵測
    results = MODEL.predict(image_path, save=False)  # 不直接保存
    
    # 獲取檔名（不含副檔名）
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    
    char_bboxes = []  # 儲存字元邊界框
    
    # 保存預測圖片和截取車牌
    for r in results:
        im_array = r.plot()  # 繪製預測結果
        cv2.imwrite(f'result/{file_name}.jpg', im_array)
        
        # 保存標籤文件並截取車牌
        if r.boxes:
            with open(f'labels/{file_name}.txt', 'w') as f:
                for box in r.boxes:
                    # 轉換為YOLO格式
                    x_center = box.xywhn[0][0].item()
                    y_center = box.xywhn[0][1].item()
                    width = box.xywhn[0][2].item()
                    height = box.xywhn[0][3].item()
                    class_id = box.cls[0].item()
                    f.write(f"{int(class_id)} {x_center} {y_center} {width} {height}\n")
                    
                    # 計算車牌在原圖中的像素座標
                    x = int((x_center - width/2) * w)
                    y = int((y_center - height/2) * h)
                    plate_w = int(width * w)
                    plate_h = int(height * h)
                    
                    # 截取車牌區域
                    plate = img[y:y+plate_h, x:x+plate_w]
                    if plate.size > 0:
                        # 保存車牌圖片
                        cv2.imwrite(f'board/{file_name}.jpg', plate)
            
            # 處理截取的車牌圖片
            plate_path = f'board/{file_name}.jpg'
            warped_path, M, rect, top_crop = process_license_plate(plate_path)
            if warped_path and M is not None:
                warped_img = cv2.imread(warped_path)
                if warped_img is None:
                    print(f"無法讀取校正後的圖片: {warped_path}")
                    continue
                # 對校正後車牌做 Tesseract 字元辨識
                custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                ocr_text = pytesseract.image_to_string(warped_img, config=custom_config)
                print(f"校正後偵測到字元: {ocr_text.strip()}")

                # 獲取字符位置
                boxes = pytesseract.image_to_boxes(warped_img, config=custom_config)
                if not boxes:
                    continue
                try:
                    M_inv = np.linalg.inv(M)
                except np.linalg.LinAlgError:
                    print(f"Cannot invert matrix M for image: {file_name}. Skipping character detection.")
                    
                    continue
                for b in boxes.splitlines():
                    b = b.split()
                    if len(b) >= 5:
                        x1_char, y1_char, x2_char, y2_char = map(int, b[1:5])
                        # Tesseract 的 y 座標從下往上
                        y1_char = warped_img.shape[0] - y1_char
                        y2_char = warped_img.shape[0] - y2_char
                        # Adjust for top crop
                        y1_char += top_crop
                        y2_char += top_crop

                        # 定義字元框的四個角點
                        pts = np.array([
                            [x1_char, y2_char],
                            [x2_char, y2_char],
                            [x2_char, y1_char],
                            [x1_char, y1_char]
                        ], dtype="float32")
                        pts = pts.reshape(-1, 1, 2)
                        original_pts = cv2.perspectiveTransform(pts, M_inv)
                        original_pts = original_pts.reshape(-1, 2).astype(int)

                        # Add offsets for the original image
                        original_pts[:,0] += x
                        original_pts[:,1] += y

                        # Clamp coordinates to image boundaries
                        x_coords = original_pts[:,0]
                        y_coords = original_pts[:,1]
                        x_min, x_max = int(x_coords.min()), int(x_coords.max())
                        y_min, y_max = int(y_coords.min()), int(y_coords.max())
                        # Clamp coordinates to image boundaries
                        x_min = max(x_min, 0)
                        y_min = max(y_min, 0)
                        x_max = min(x_max, img.shape[1] - 1)
                        y_max = min(y_max, img.shape[0] - 1)
                        char_bboxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
                        
                        # 在原圖上繪製字符框
                        cv2.rectangle(img, 
                                      (x_min, y_min),
                                      (x_max, y_max),
                                      (0, 255, 0), 2)
    
    # 儲存標註後的原圖
    cv2.imwrite(f'number/{file_name}.jpg', img)

    # Write results to the output txt file in the specified format
    if char_bboxes:
        with open(output_file, 'a') as f:
            f.write(f"{file_name}.jpg\n")
            f.write(f"{len(char_bboxes)}\n")
            for bbox in char_bboxes:
                f.write(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
    else:
        print(f"No characters detected for image: {file_name}")

def main():
    # Initialize the output file with the group leader's student ID
    global output_file
    output_file = '411185011.txt'  # Replace with the actual student ID
    with open(output_file, 'w') as f:
        f.write("")

    folder_path = 'sample 2024'
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print()
        print(f"Processing: {image_file}")
        process_image(image_path)

if __name__ == "__main__":
    main()

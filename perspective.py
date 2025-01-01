import cv2
import numpy as np
from os import makedirs, path

def process_license_plate(image_path, debug=False):
    # 建立輸出目錄（若不存在）
    makedirs('processed_board', exist_ok=True)

    # 讀取影像
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取圖片: {image_path}")
        return None, None, None, None

    # 轉換為灰度圖並進行預處理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Enhance contrast
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Moderate blur size

    license_plate_contour = None
    max_area = 0

    for (canny_low, canny_high) in [(30, 100), (60, 170), (90, 220)]:  # Adjusted thresholds
        edges = cv2.Canny(blur, canny_low, canny_high)
        kernel = np.ones((5, 5), np.uint8)  # Increased kernel size
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            if len(approx) == 4 and area > 500 and area > max_area:
                license_plate_contour = approx
                max_area = area
        if license_plate_contour is not None:
            break  # Stop once a valid plate is found

    if license_plate_contour is None:
        # 若無法找到車牌，使用圖片邊緣作為邊界
        h, w = image.shape[:2]
        license_plate_contour = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype="float32")

    # 獲取車牌的四個角點
    pts = license_plate_contour.reshape(4, 2)

    # 將角點排序：上左、上右、下右、下左
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # 計算車牌的寬度和高度
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    # Maintain aspect ratio if necessary
    aspect_ratio = maxWidth / maxHeight
    desired_aspect = 4.0 / 1.0  # Typical license plate aspect ratio
    if aspect_ratio > desired_aspect:
        maxWidth = int(maxHeight * desired_aspect)
    else:
        maxHeight = int(maxWidth / desired_aspect)

    # 設定目標點，對應於校正後的車牌影像
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # 計算透視變換矩陣
    M = cv2.getPerspectiveTransform(rect, dst)
    # 進行透視變換
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Crop top and bottom 10%
    height = warped.shape[0]
    top_crop = int(0.15 * height)
    bottom_crop = int(0.12 * height)
    warped = warped[top_crop:height - bottom_crop, :]

    # 儲存結果
    output_path = path.join('processed_board', path.basename(image_path))
    cv2.imwrite(output_path, warped)

    # 若啟用除錯模式，顯示結果
    if debug:
        cv2.imshow('Original Image', image)
        cv2.imshow('Detected License Plate', warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output_path, M, rect, top_crop

if __name__ == '__main__':
    # 執行車牌處理
    image_path = 'board/004.jpg'  # 使用相對路徑
    output_path, M, rect, top_crop = process_license_plate(image_path, debug=True)
    if output_path and M is not None:
        print(f"車牌處理完成，結果儲存於 {output_path}")
    else:
        print("車牌處理失敗")

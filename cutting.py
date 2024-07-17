import os
import shutil
import cv2
import dlib
from PIL import Image, ImageEnhance

detector = dlib.get_frontal_face_detector() # 載入 dlib 的臉部檢測器

def ClearDirectory(directory):
    # 先清空 "Real-ESRGAN/inputs 目錄下的所有檔案與資料夾
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # 刪除檔案或符號連結
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 刪除資料夾
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def Detect_and_crop_face(output_size):
    file_names = os.listdir("input")
    file_names = [f for f in file_names if os.path.isfile(os.path.join("input", f))]
    image_path = os.path.join("input", file_names[0])

    img = cv2.imread(image_path)     # 讀取圖片
    if img is None:
        print("Image not found!")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 轉成灰階
    faces = detector(gray) # 檢測人臉
    if len(faces) == 0:
        print("No face detected!") 
        return
    
    face = faces[0] # 只處理第一個檢測到的人臉
    x, y, w, h = (face.left(), face.top(), face.width(), face.height()) # 計算人臉區域
    padding = int(0.5 * h)  # 擴展高度的一半作為以容納軀幹
    cropped_img = img[max(0, y-padding):min(img.shape[0], y+h+padding), max(0, x-padding):min(img.shape[1], x+w+padding)] # 裁剪人臉區域
    
    pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))  # 轉換為PIL圖片
    pil_img = pil_img.resize(output_size, Image.Resampling.LANCZOS) # 調整圖片大小
    enhancer = ImageEnhance.Sharpness(pil_img) # 簡單的畫質增強
    pil_img = enhancer.enhance(2.0)  # 增強銳度
    pil_img.save("mid/image.jpg")

    ClearDirectory('Real-ESRGAN/inputs')  # 先清空上一次的輸出目錄

    output_path = "Real-ESRGAN/inputs/image.jpg" # 儲存結果
    pil_img.save("Real-ESRGAN/inputs/image.jpg")
    print(f"Processed image saved as {output_path}")

Detect_and_crop_face((500, 500))



import cv2

def ScalingImage(output_size):
    print("Scaling image...")
    # 讀取圖片
    img = cv2.imread('Real-ESRGAN/results/image_out.jpg')

    # 縮放圖片到 500x500
    resized_img = cv2.resize(img, output_size)

    # 保存縮放後的圖片
    cv2.imwrite('output/image_out.jpg', resized_img)
    print(f"Processed image saved as 'output/image_out.jpg'")

ScalingImage((500, 500))



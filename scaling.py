import cv2

def ScalingImage(output_size):
    # 讀取圖片
    img = cv2.imread('Real-ESRGAN/results/image_out.jpg')

    # 縮放圖片到 500x500
    resized_img = cv2.resize(img, (500, 500))

    # 保存縮放後的圖片
    cv2.imwrite('output/image.jpg', resized_img)

ScalingImage((500, 500))



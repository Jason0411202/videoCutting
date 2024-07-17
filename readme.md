# 環境配置
* 環境: python 3.11
* 註: win10 安裝 dlib   https://github.com/z-mahmud22/Dlib_Windows_Python3.x

```
conda create -n real-esrgan python=3.11
conda activate real-esrgan
git clone https://github.com/xinntao/Real-ESRGAN.git
cd Real-ESRGAN
pip install --verbose basicsr --use-pep517
pip install facexlib
pip install gfpgan
pip install -r requirements.txt
python setup.py develop
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P experiments/pretrained_models
```

## 測試 real-esrgan 是否正常運作
在 Real-ESRGAN 專案根目錄下執行以下指令
```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i inputs --face_enhance
```

## 除錯紀錄
* 有時候會出現以下錯誤 `No module named ‘torchvision.transforms.functional_tensor`
    * 解決方法: https://blog.csdn.net/lanxing147/article/details/136625264

# 執行
輸入的圖片放在 input 資料夾中
圖片將會輸出在 output 資料夾中
```bash
python main.py
```
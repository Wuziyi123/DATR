import pywt
import numpy as np
import cv2
import matplotlib.pyplot as plt

def wavelet_enhance(image_path, scale=2.0, threshold=0.05, wavelet='db4', levels=3):
    # 读取图像
    image = cv2.imread(image_path, 0).astype(np.float32) / 255.0

    # 小波分解
    coeffs = pywt.wavedec2(image, wavelet, level=levels)
    LL, (LH, HL, HH) = coeffs[0], coeffs[1]

    # 增强最细尺度高频分量
    HH_enhanced = HH * scale
    HL_enhanced = HL * scale
    LH_enhanced = LH * scale

    # 软阈值去噪
    HH_enhanced = pywt.threshold(HH_enhanced, threshold, mode='soft')
    HL_enhanced = pywt.threshold(HL_enhanced, threshold, mode='soft')
    LH_enhanced = pywt.threshold(LH_enhanced, threshold, mode='soft')

    # 替换高频分量并重构
    enhanced_coeffs = (LL, (LH_enhanced, HL_enhanced, HH_enhanced))
    enhanced_image = pywt.waverec2(enhanced_coeffs, wavelet)
    enhanced_image = np.clip(enhanced_image, 0, 1)

    return enhanced_image

# 使用示例
enhanced = wavelet_enhance('images/ILSVRC2012.JPEG', scale=2.0, threshold=0.05)

# 显示结果
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(cv2.imread('images/ILSVRC2012.JPEG'), cmap='gray'), plt.title('Original')
plt.subplot(122), plt.imshow(enhanced, cmap='gray'), plt.title('Enhanced')
plt.show()
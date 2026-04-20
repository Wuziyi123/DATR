import cv2
import numpy as np


def enhance_color_contrast(img):
    # RGB转Lab空间
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # CLAHE增强色度通道（a和b通道）
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    a = clahe.apply(a)
    b = clahe.apply(b)

    # 合并通道并转回BGR
    enhanced_lab = cv2.merge([l, a, b])
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_img


# 使用示例
image = cv2.imread("images/ILSVRC2012.JPEG")
enhanced_image = enhance_color_contrast(image)

from skimage.segmentation import slic


def multi_feature_saliency(img):
    # SLIC超像素分割
    segments = slic(img, n_segments=300, compactness=20)

    # 计算每个超像素的Lab均值
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    features = []
    for label in np.unique(segments):
        mask = (segments == label).astype(np.uint8)
        mean_color =  np.array(cv2.mean(lab, mask)[:3])
        features.append(mean_color)

    # 将列表转换为NumPy数组
    features = np.array(features)  # 现在是一个二维数组 [N, 3]
    # 计算颜色对比度显著性
    saliency = np.zeros_like(img[:, :, 0], dtype=np.float32)
    for i in range(len(features)):
        diff = np.linalg.norm(features - features[i], axis=1)
        saliency[segments == i] = np.mean(diff)

    # 归一化显著性图
    saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX)
    return saliency.astype(np.uint8)


# 生成显著性图
saliency_map = multi_feature_saliency(enhanced_image)
cv2.imwrite("saliency_map-1.jpg", saliency_map)
import cv2
import numpy as np


def auto_canny_edge_detection(image, sigma=0.33):
    md = np.median(image)
    lower_value = int(max(0, (1.0-sigma) * md))
    upper_value = int(min(255, (1.0+sigma) * md))
    return cv2.Canny(image, lower_value, upper_value)


# 图像读取与预处理
# image = cv2.imread('images/ILSVRC2012.JPEG')
image = cv2.imread('images/2025.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = auto_canny_edge_detection(blurred)
# edges = cv2.Canny(blurred, 50, 150)
# cv2.imshow('edges', edges)
# cv2.waitKey(0)
# --------------------------------------------------------------------


# 形态学闭运算（填充边缘内部）
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


# 连通域分析（保留面积较大的边缘区域）
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_edges)
min_area = 100  # 根据图像调整阈值
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] < min_area:
        closed_edges[labels == i] = 0
cv2.imwrite("closed_edges.jpg", closed_edges)


# 示例：基于OpenCV的显著性检测（简单方法）
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliency_map) = saliency.computeSaliency(image)
saliency_map = (saliency_map * 255).astype("uint8")
# _, binary = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("saliency_map.jpg", saliency_map)



high_saliency = np.where(saliency_map > 200)
input_points = np.column_stack((high_saliency[1], high_saliency[0]))  # 提取高显著点坐标


import matplotlib.pyplot as plt
plt.figure(figsize=(15, 5))

# 子图1：原始图像+采样点
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.scatter(input_points[:, 0], input_points[:, 1], c='red', s=10)
plt.title("Original Image with Points")

# 子图2：显著图热图（网页2的热图方法）
plt.subplot(1, 2, 2)
plt.imshow(saliency_map, cmap='jet')
plt.title("Saliency Heatmap")

# 子图3：显著区域二值掩膜
# plt.subplot(1, 3, 3)
# _, binary_mask = cv2.threshold(saliency_map, 200, 255, cv2.THRESH_BINARY)
# plt.imshow(binary_mask, cmap='gray')
# plt.title("Binary Saliency Mask")

plt.tight_layout()
plt.show()



# 方法2示例：用边缘图作为掩膜，过滤显著性区域
# 将边缘图转为二值掩膜
edge_mask = cv2.bitwise_not(closed_edges)  # 边缘为黑色，背景为白色
# 对显著性图应用掩膜
saliency_masked = cv2.bitwise_and(saliency_map, saliency_map, mask=edge_mask)
# 二值化并提取显著区域
_, binary_saliency = cv2.threshold(saliency_masked, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("binary_saliency.jpg", binary_saliency)




# 步骤1-4：生成cleaned_saliency（略，参考之前的代码）

# 步骤5：提取多个显著对象
# --- 形态学开运算优化分割 ---
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
cleaned_saliency = cv2.morphologyEx(binary_saliency, cv2.MORPH_OPEN, kernel)


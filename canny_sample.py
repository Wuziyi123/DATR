import cv2
import numpy as np


# 图像读取与预处理
# image = cv2.imread('images/ILSVRC2012.JPEG')
image = cv2.imread('images/2025.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# 示例：基于OpenCV的显著性检测（简单方法）
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliency_map) = saliency.computeSaliency(image)
saliency_map = (saliency_map * 255).astype("uint8")
# _, binary = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("saliency_map.jpg", saliency_map)
high_saliency = np.where(saliency_map > 180)
input_points = np.column_stack((high_saliency[1], high_saliency[0]))  # 提取高显著点坐标
scores = saliency_map[input_points[:, 1], input_points[:, 0]]
scores = scores.flatten() # 确保是一维数组


def grid_sampling_on_image(image_path, points, scores, grid_size=8, target_num=50):
    # 读取图像
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # 网格划分
    cell_width = w / grid_size
    cell_height = h / grid_size

    selected_points = []

    for i in range(grid_size):
        for j in range(grid_size):
            x_min = int(i * cell_width)
            x_max = int((i + 1) * cell_width)
            y_min = int(j * cell_height)
            y_max = int((j + 1) * cell_height)

            # 筛选当前网格内的点
            mask = (points[:, 0] >= x_min) & (points[:, 0] < x_max) & \
                   (points[:, 1] >= y_min) & (points[:, 1] < y_max)
            if np.any(mask):
                # 选择显著度最高的点
                local_scores = scores[mask]
                max_idx = np.argmax(local_scores)
                selected_point = points[mask][max_idx]
                selected_points.append(selected_point)

    # 转换为整数坐标
    selected_points = np.array(selected_points, dtype=int)

    # 绘制在图像上
    for (x, y) in selected_points:
        cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # 红色点

    # 显示或保存图像
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 使用示例
grid_sampling_on_image('images/2025.jpg', input_points, scores, grid_size=8)

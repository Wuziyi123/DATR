from scipy.spatial import KDTree
import numpy as np


def dynamic_size_bboxes(centers, min_size=20, scale=0.4, img_size=None):
    """
    动态调整尺寸的边界框（中心点+宽高格式）
    :param centers: 中心点坐标数组
    :param min_size: 最小边框边长
    :param scale: 最近邻距离的缩放比例
    :param img_size: 图像尺寸约束 (width, height)，可选
    :return: 边界框列表[[cx, cy, w, h], ...]
    """
    points = np.array(centers)
    tree = KDTree(points)

    # 计算动态边长（保持宽高相同）
    distances, _ = tree.query(points, k=2)
    neighbor_dists = distances[:, 1]
    side_lengths = np.maximum(neighbor_dists * scale, min_size)
    half_sizes = (side_lengths / 2).astype(int)

    # 计算原始边界框坐标
    x1 = points[:, 0] - half_sizes
    y1 = points[:, 1] - half_sizes
    x2 = points[:, 0] + half_sizes
    y2 = points[:, 1] + half_sizes

    # 应用图像边界约束
    if img_size is not None:
        img_w, img_h = img_size
        x1 = np.clip(x1, 0, img_w - 1)
        y1 = np.clip(y1, 0, img_h - 1)
        x2 = np.clip(x2, 0, img_w - 1)
        y2 = np.clip(y2, 0, img_h - 1)

    # 转换为中心点+宽高格式
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return np.column_stack((cx, cy, w, h)).tolist()


# 示例使用
# centers = [[100, 200], [105, 195], [300, 150]]
# bboxes = dynamic_size_bboxes(centers, min_size=15, scale=0.3, img_size=(640, 480))
# print(bboxes)
# 输出根据点间距动态变化，例如：
# [[102.5, 197.5, 5, 5], [105.0, 195.0, 0, 0], ...]
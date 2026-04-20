import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import torch
from edge_sam import SamPredictor, sam_model_registry
import torchvision.transforms as T
from skimage.transform import resize
from skimage.measure import label
from PIL import Image
from skimage.measure import label, regionprops


def random_crop(image: Image.Image, alpha: float = 0.1) -> Image.Image:
    """Randomly crops an image within a size range determined by alpha and the image dimensions.

    Args:
        image (Image): The input image to crop.
        alpha (float): The minimum scale factor for the crop as a proportion of the smallest dimension.

    Returns:
        PIL Image or Tensor: Cropped image
    """
    # Get the width and height of the original image
    w, h = image.size
    # Determine the size of the crop based on alpha and the smallest dimension
    n_px = np.random.uniform(low=alpha, high=0.9) * min(h, w)
    # Perform the crop
    cropped = T.RandomCrop(int(n_px))(image)

    return cropped


def grid_sampling_on_image(image_size, points, scores, grid_size=8, target_num=50):
    # 读取图像
    h, w = image_size
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
    return selected_points


def get_points(image):
    # image = cv2.imread(path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 示例：基于OpenCV的显著性检测（简单方法）
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliency_map) = saliency.computeSaliency(image)
    saliency_map = (saliency_map * 255).astype("uint8")
    # cv2.imwrite("saliency_map.jpg", saliency_map)
    high_saliency = np.where(saliency_map > 180)
    input_points = np.column_stack((high_saliency[1], high_saliency[0]))  # 提取高显著点坐标
    scores = saliency_map[input_points[:, 1], input_points[:, 0]]
    scores = scores.flatten()  # 确保是一维数组
    size = (image.shape[:2])
    selected_points = grid_sampling_on_image(size,
                                         input_points, scores, grid_size=8)
    return selected_points


import numpy as np
from typing import List, Tuple

from typing import Tuple, List
import numpy as np


def adaptive_crop(
        image_size: Tuple[int, int],
        mask: np.ndarray,
        points: np.ndarray,
        num_crops: int = 20,
        max_attempts: int = 200
) -> List[Tuple[int, int, int, int]]:
    """
    Args:
        image_size: (height, width) 原图尺寸
        mask: 二维掩码矩阵（0/255）
        points: (N,2) 对象分布点坐标
        num_crops: 需要生成的裁剪数量
        max_attempts: 单尺度最大尝试次数

    Returns:
        List of (x, y, w, h) 正方形裁剪框
    """
    h, w = image_size
    length = min(h, w)  # 基准边长
    crop_boxes = []

    # 预处理掩码区域
    mask_area = np.sum(mask > 0)
    valid_mask = mask_area > 0.02 * h * w

    # 尺度划分（动态适应图像尺寸）
    scale_levels = [
        (0.5, 0.9),  # 大尺度
        # (0.2, 0.5),  # 中尺度
        # (0.05, 0.2)  # 小尺度
    ]

    # 生成候选框主逻辑
    for scale_min, scale_max in scale_levels:
        attempts = 0
        while len(crop_boxes) < num_crops and attempts < max_attempts:
            attempts += 1

            # 随机选择尺度并计算正方形边长
            scale = np.random.uniform(scale_min, scale_max)
            crop_size = int(length * scale)

            # 生成候选框中心
            if valid_mask:
                # 获取所有满足条件的掩码点坐标
                y_indices, x_indices = np.where(
                    (mask > 0) &
                    (np.arange(h)[:, None] >= crop_size // 2) &
                    (np.arange(h)[:, None] <= h - crop_size // 2) &
                    (np.arange(w)[None, :] >= crop_size // 2) &
                    (np.arange(w)[None, :] <= w - crop_size // 2)
                )

                if len(y_indices) == 0:
                    continue  # 无有效区域时跳过

                idx = np.random.choice(len(y_indices))
                center_y = y_indices[idx]
                center_x = x_indices[idx]
            else:
                # 在有效区域内生成中心点
                x_min = crop_size // 2
                x_max = w - crop_size // 2
                y_min = crop_size // 2
                y_max = h - crop_size // 2

                if x_max <= x_min or y_max <= y_min:
                    continue  # 无法生成有效中心点时跳过

                center_x = np.random.randint(x_min, x_max)
                center_y = np.random.randint(y_min, y_max)

            # 计算边界（确保正方形不越界）
            y1 = center_y - crop_size // 2
            x1 = center_x - crop_size // 2
            y2 = y1 + crop_size
            x2 = x1 + crop_size

            # 有效性验证（至少包含一个对象点）
            if len(points) > 0:
                in_box = (points[:, 0] >= y1) & (points[:, 0] <= y2) & \
                         (points[:, 1] >= x1) & (points[:, 1] <= x2)
                if np.sum(in_box) < 1:
                    continue

            crop_boxes.append((x1, y1, crop_size, crop_size))

    # 补足剩余数量（强制生成完整正方形）
    while len(crop_boxes) < num_crops:
        min_size = max(1, int(0.05 * length))
        max_size = max(min_size + 1, int(0.2 * length))
        crop_size = np.random.randint(min_size, max_size)
        crop_size = min(crop_size, w, h)  # 最终尺寸不超过原图

        try:
            x1 = np.random.randint(0, w - crop_size + 1)
            y1 = np.random.randint(0, h - crop_size + 1)
        except ValueError:
            continue  # 处理极端小尺寸情况

        crop_boxes.append((x1, y1, crop_size, crop_size))

    return crop_boxes[:num_crops]


#*************************************************************************************

def get_crop_Images(paths):
    batch_selected_points = []
    for i in range(len(paths)):
        path = paths[i]
        image = cv2.imread(path)
        selected_points = get_points(image)
        batch_selected_points.append(selected_points)
    return batch_selected_points

        # masks = [1,565,800]
        # masks, scores, logits = predictor.predict(
        #     point_coords=np.array(selected_points),
        #     point_labels=np.ones(len(selected_points), dtype=np.int64),
        #     num_multimask_outputs=1,
        #     use_stability_score=True
        # )
        # mask = (masks[0] > 0.5)  # 示例布尔掩码
        # resized_array = resize(mask, (224, 224), order=1, anti_aliasing=False)
        # batch_masks.append(resized_array)
    # tensors = [torch.from_numpy(arr) for arr in batch_masks]
    # mask_imgs = torch.stack(tensors)

        # 使用示例x
        # img = cv2.imread(path)  # OpenCV读取图像
        # 生成裁剪区域

    #     crop_regions = adaptive_crop(image_size, masks[0], selected_points, n_samples)
    #     for i, (x, y, w, h) in enumerate(crop_regions):
    #         crop = image[y:y + h, x:x + w]
    #         crop = processor(Image.fromarray(crop))
    #         crop_imgs.append(crop)
    #     crop_imgs = torch.stack(crop_imgs, dim=0)
    #     batch_crop_imgs.append(crop_imgs)
    # del crop_imgs, crop
    # torch.cuda.empty_cache()
    # return torch.stack(batch_crop_imgs, dim=0)


# if __name__ == "__main__":
#     sam = sam_model_registry["edge_sam"](checkpoint="edge_sam_3x.pth")
#     sam.to(device="cuda")
#     sam.eval()
#     predictor = SamPredictor(sam)
#     crops = get_crop_Images(predictor)
#     pass

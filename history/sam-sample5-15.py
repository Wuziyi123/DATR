import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
import torch
from edge_sam import SamPredictor, sam_model_registry
import torchvision.transforms as T
from skimage.measure import label
from PIL import Image
from skimage.measure import label, regionprops
from clip.clip import _transform
from typing import List, Tuple


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    e = pos_points[:, 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def crop_roi(image, mask):
    # 根据掩码最小外接矩形裁剪区域
    y_indices, x_indices = np.where(mask)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    cropped = image.crop((x_min, y_min, x_max, y_max))
    return T.Resize((224,224))(cropped)  # 调整回CLIP输入尺寸


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
    # image = cv2.imread(image_path)
    # h, w = image.shape[:2]
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



def adaptive_crop(
        image_size: Tuple[int, int],
        mask: np.ndarray,
        path: str,
        points: np.ndarray,
        num_crops: int = 20,
        max_attempts: int = 200
) -> List[Tuple[int, int, int, int]]:
    """
    生成正方形的自适应裁剪框

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
    crop_boxes = []

    # 预处理掩码区域
    mask_area = np.sum(mask > 0)
    valid_mask = mask_area > 0.02 * h * w

    # 正方形尺度范围（取高度和宽度的最小值作为基准）
    scale_levels = [
        (0.1, 0.9),  # 大尺度（边长占min_dim的50%-90%）
        # (0.2, 0.5),  # 中尺度
        # (0.05, 0.2)  # 小尺度
    ]

    # 生成候选框主逻辑
    for scale_min, scale_max in scale_levels:
        attempts = 0
        while len(crop_boxes) < num_crops and attempts < max_attempts:
            attempts += 1

            # 随机选择正方形边长（基于图像短边）
            crop_size = int(np.random.uniform(scale_min, scale_max) * min(h, w))

            # 生成中心点（优先掩码区域）
            if valid_mask:
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) == 0:
                    valid_mask = False
                    continue
                idx = np.random.choice(len(y_indices))
                center_y = y_indices[idx]
                center_x = x_indices[idx]
            else:
                if len(points) > 0:
                    pt = points[np.random.randint(len(points))]
                    center_y, center_x = pt[0], pt[1]
                else:
                    center_y = np.random.randint(0, h)
                    center_x = np.random.randint(0, w)

            # 计算正方形边界（确保不越界）
            y1 = max(0, center_y - crop_size // 2)
            x1 = max(0, center_x - crop_size // 2)
            y2 = y1 + crop_size
            x2 = x1 + crop_size

            # 越界修正
            if y2 > h:
                y1 = max(0, h - crop_size)
                y2 = h
            if x2 > w:
                x1 = max(0, w - crop_size)
                x2 = w

            # 有效性验证（至少包含一个分布点）
            if len(points) > 0:
                in_box = (points[:, 0] >= y1) & (points[:, 0] <= y2) & \
                         (points[:, 1] >= x1) & (points[:, 1] <= x2)
                if np.sum(in_box) < 1:
                    continue

            crop_boxes.append((x1, y1, x2 - x1, y2 - y1))

    # 补足剩余数量（随机小正方形）
    while len(crop_boxes) < num_crops:
        image = Image.open(path)
        crop_size = int(np.random.uniform(low=0.1, high=0.9) * min(h, w))
        # Perform the crop
        cropped = T.RandomCrop(int(crop_size))(image)
        crop_boxes.append((x1, y1, crop_size, crop_size))

    return crop_boxes[:num_crops]


#*************************************************************************************

def get_crop_Images(paths, processor=_transform(224), n_samples=20, predictor=None):
    batch_crop_imgs = []
    device = "cuda"
    for i in range(len(paths)):
        crop_imgs = []
        path = paths[i]
        image = cv2.imread(path)
        image_size = image.shape[:2]
        image_plt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # sam = sam_model_registry["edge_sam"](checkpoint="edge_sam_3x.pth")
        # sam.to(device)
        # predictor = SamPredictor(sam)

        predictor.set_image(image_plt)
        selected_points = get_points(image)

        # masks = [1,565,800]
        masks, scores, logits = predictor.predict(
            point_coords=np.array(selected_points),
            point_labels=np.ones(len(selected_points), dtype=np.int64),
            num_multimask_outputs=1,
            use_stability_score=True
        )

        # 使用示例
        # img = cv2.imread(path)  # OpenCV读取图像
        # 生成裁剪区域
        # img = cv2.imread(path)  # OpenCV读取图像

        img1 = Image.open(path)
        w, h = img1.size
        # 生成裁剪区域
        for _ in range(n_samples):
            crop_size = np.random.uniform(low=0.1, high=0.9) * min(h, w)
            crop = T.RandomCrop(int(crop_size))(img1)
            crop = processor(crop)
            crop_imgs.append(crop)
        crop_imgs = torch.stack(crop_imgs, dim=0)
        batch_crop_imgs.append(crop_imgs)

        # crop_regions = adaptive_crop(image_size, masks[0], path, selected_points, n_samples)

        # for i, (mask, score) in enumerate(zip(masks, scores)):
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(image_plt)
        #     show_mask(mask, plt.gca())
        #     # show_points(selected_points, np.array([1]), plt.gca())
        #     plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        #     plt.axis('off')
        #     plt.show()
        #
        # # 可视化展示
        # plt.figure(figsize=(15, 10))
        # for i, (x, y, w, h) in enumerate(crop_regions):
        #     crop = img[y:y + h, x:x + w]
        #     plt.subplot(4, 5, i + 1)
        #     plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        #     plt.axis('off')
        # plt.tight_layout()
        # plt.show()
        # exit(0)


        # for i, (x, y, w, h) in enumerate(crop_regions):
        #     crop = image[y:y + h, x:x + w]
        #     crop = processor(Image.fromarray(crop))
        #     crop_imgs.append(crop)
        # crop_imgs = torch.stack(crop_imgs, dim=0)
        # batch_crop_imgs.append(crop_imgs)

    return torch.stack(batch_crop_imgs, dim=0)


if __name__ == "__main__":
    sam = sam_model_registry["edge_sam"](checkpoint="edge_sam_3x.pth")
    sam.to(device="cuda")
    sam.eval()
    predictor = SamPredictor(sam)
    crops = get_crop_Images(['run.jpg',])
    pass

import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from edge_sam import SamPredictor, sam_model_registry
import torchvision.transforms as T
from skimage.measure import label


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
    return selected_points


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


def get_points(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_plt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 示例：基于OpenCV的显著性检测（简单方法）
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    (success, saliency_map) = saliency.computeSaliency(image)
    saliency_map = (saliency_map * 255).astype("uint8")
    # cv2.imwrite("saliency_map.jpg", saliency_map)
    high_saliency = np.where(saliency_map > 160)
    input_points = np.column_stack((high_saliency[1], high_saliency[0]))  # 提取高显著点坐标
    scores = saliency_map[input_points[:, 1], input_points[:, 0]]
    scores = scores.flatten()  # 确保是一维数组
    selected_points = grid_sampling_on_image(path,
                                         input_points, scores, grid_size=8)
    return selected_points


def adaptive_crop(img, mask, num_crops=20):
    # 参数初始化
    h, w = img.shape[:2]
    scales = [3/4, 1/2, 1/4, 1/8]
    crops = []
    used_boxes = []

    # 生成连通域标签
    labeled_mask = label(mask)

    while len(crops) < num_crops:
        # 随机选择尺度
        scale = np.random.choice(scales)
        crop_w, crop_h = int(w * scale), int(h * scale)

        # 生成候选区域
        max_attempts = 100
        for _ in range(max_attempts):
            # 随机起始点
            x = np.random.randint(0, w - crop_w)
            y = np.random.randint(0, h - crop_h)

            # 有效性验证
            region = labeled_mask[y:y + crop_h, x:x + crop_w]
            if np.any(region > 0):
                # 多样性检查
                current_box = (x, y, x + crop_w, y + crop_h)
                if not is_overlapping(current_box, used_boxes):
                    crops.append((x, y, crop_w, crop_h))
                    used_boxes.append(current_box)
                    break

    return crops


def is_overlapping(new_box, existing_boxes, iou_thresh=0.3):
    # IoU计算
    x1, y1, x2, y2 = new_box
    area_new = (x2 - x1) * (y2 - y1)

    for box in existing_boxes:
        xi1 = max(x1, box[0])
        yi1 = max(y1, box[1])
        xi2 = min(x2, box[2])
        yi2 = min(y2, box[3])

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        iou = inter_area / (area_new + (box[2] - box[0]) * (box[3] - box[1]) - inter_area)

        if iou > iou_thresh:
            return True
    return False


#*************************************************************************************

def get_crop_Images(path="run.jpg",):
    device = "cuda"
    image = cv2.imread(path)
    image_plt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam = sam_model_registry["edge_sam"](checkpoint="edge_sam_3x.pth")
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_plt)
    selected_points = get_points(path)

    # masks = [1,565,800]
    masks, scores, logits = predictor.predict(
        point_coords=np.array(selected_points),
        point_labels=np.ones(len(selected_points), dtype=np.int64),
        num_multimask_outputs=1,
        use_stability_score=True
    )

    # 使用示例
    img = cv2.imread(path)  # OpenCV读取图像
    # 生成裁剪区域
    crop_regions = adaptive_crop(img, masks[0])

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image_plt)
        show_mask(mask, plt.gca())
        # show_points(selected_points, np.array([1]), plt.gca())
        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

    # 可视化展示
    plt.figure(figsize=(15, 10))
    for i, (x, y, w, h) in enumerate(crop_regions):
        crop = img[y:y + h, x:x + w]
        plt.subplot(4, 5, i + 1)
        plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    return crop_regions


# for i, (mask, score) in enumerate(zip(masks, scores)):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image_plt)
#     show_mask(mask, plt.gca())
#     # show_points(selected_points, np.array([1]), plt.gca())
#     plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
#     plt.axis('off')
#     plt.show()

# pil_image = Image.fromarray(img_array)
# plt.figure(figsize=(15, 10))
# for i, (x, y, w, h) in enumerate(crop_regions):
#     crop = img[y:y + h, x:x + w]
#     plt.subplot(4, 5, i + 1)
#     plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
# plt.tight_layout()
# plt.show()

if __name__ == "__main__":
    crops = get_crop_Images()
    pass

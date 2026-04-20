import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from PIL import Image, ImageDraw
import numpy as np
import json
import os
from tqdm import tqdm
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt  # 添加可视化库
import matplotlib.patches as patches  # 用于绘制边界框


def visualize_proposals(image, proposals, num_boxes=5):
    """
    可视化原始图像和提案边界框

    参数:
        image: PIL.Image - 原始图像
        proposals: list - 边界框列表 [x_min, y_min, x_max, y_max]
        num_boxes: int - 要显示的边界框数量
    """
    # 创建绘图对象
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    # 绘制前N个建议框
    for i, box in enumerate(proposals[:num_boxes]):
        # 提取坐标
        x_min, y_min, x_max, y_max = box

        # 创建矩形框
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor=plt.cm.hsv(i / num_boxes),  # 使用彩虹色系区分不同框
            facecolor='none'
        )

        # 添加到图像
        ax.add_patch(rect)

        # 添加序号标签
        ax.text(
            x_min + 5,
            y_min + 5,
            str(i + 1),
            fontsize=12,
            color='white',
            bbox=dict(facecolor=plt.cm.hsv(i / num_boxes), alpha=0.8, edgecolor='none', pad=0)
        )

    plt.axis('off')
    plt.title(f"Top-{num_boxes} RPN Proposals")
    plt.show()


def generate_rpn_proposals(image_dir, output_file, num_proposals=80):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 构建模型时指定RPN参数（关键改进）
    model = fasterrcnn_resnet50_fpn(
        pretrained=True,
        rpn_post_nms_top_n_test=num_proposals,  # 控制输出提案数量
        box_score_thresh=0.05,  # 分数阈值
        box_nms_thresh=0.7  # NMS阈值
    ).to(device)
    model.eval()

    # 获取内置转换器（关键改进）
    transform = model.transform

    proposals_dict = {}
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 添加可视化标志
    visualize_done = False

    for img_idx, img_file in enumerate(tqdm(image_files)):
        img_path = os.path.join(image_dir, img_file)
        image_id = os.path.splitext(img_file)[0]

        try:
            # 读取原始图像并记录尺寸
            orig_image = Image.open(img_path).convert('RGB')
            orig_size = orig_image.size  # (width, height)

            # 使用内置转换器预处理（核心改进）
            img_tensor = F.to_tensor(orig_image).to(device)
            images_transformed, _ = transform([img_tensor])  # 应用内置预处理

            # 获取预处理后的图像尺寸信息
            resized_size = images_transformed.image_sizes[0]  # (height, width)

            # 提取特征并生成提案
            with torch.no_grad():
                features = model.backbone(images_transformed.tensors)
                proposals, _ = model.rpn(images_transformed, features)

            # 获取当前图像的提案（已排序和过滤）
            img_proposals = proposals[0].cpu().numpy()

            # 转换坐标回原始尺寸（关键改进）
            # 计算宽高缩放因子（考虑填充影响）
            scale_x = orig_size[0] / resized_size[1]  # 原始宽 / 缩放后宽
            scale_y = orig_size[1] / resized_size[0]  # 原始高 / 缩放后高

            # 应用缩放因子转换坐标
            img_proposals[:, [0, 2]] *= scale_x
            img_proposals[:, [1, 3]] *= scale_y

            # 确保坐标在图像范围内
            img_proposals = img_proposals.clip(0, [orig_size[0], orig_size[1],
                                                   orig_size[0], orig_size[1]])

            proposals_dict[image_id] = img_proposals.tolist()

            # ========= 新增：处理第一张图片时进行可视化 =========
            if img_idx == 2 and not visualize_done:
                print("\n[可视化] 正在显示第一张图片的建议框...")
                print(f"图像: {img_file}")
                print(f"原始尺寸: {orig_size[0]}x{orig_size[1]}")
                print(f"处理后尺寸: {resized_size[1]}x{resized_size[0]}")
                print(f"生成建议框数量: {len(img_proposals)}")
                print("前10个建议框坐标:")
                for i, box in enumerate(img_proposals[:5]):
                    print(f"Box {i + 1}: {box}")

                # 可视化前5个建议框
                visualize_proposals(orig_image, img_proposals, num_boxes=5)
                visualize_done = True  # 确保只可视化一次

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            proposals_dict[image_id] = []

    # 保存到文件
    with open(output_file, 'w') as f:
        json.dump(proposals_dict, f)
    print(f"Saved proposals to {output_file}")


if __name__ == "__main__":
    image_dir = '/home/wuziyi/Project/Flickr/Flickr30K_Entities/data/flickr30k-images'
    output_file = '/home/wuziyi/Project/Flickr/Flickr30K_Entities/flickr30k_rpn_proposals-U.json'
    generate_rpn_proposals(image_dir, output_file)
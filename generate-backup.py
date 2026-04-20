import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from PIL import Image
import numpy as np
import json
import os
from tqdm import tqdm
import torchvision.transforms.functional as F


def generate_rpn_proposals(image_dir, output_file, num_proposals=50):
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

    for img_file in tqdm(image_files):
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
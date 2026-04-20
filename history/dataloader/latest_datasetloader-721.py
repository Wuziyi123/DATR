import pandas as pd
import torch
import torch.nn as nn
import torch.fft
import pywt
import numpy as np
from transformers import BertModel, BertConfig, BertTokenizer
from timm.models.vision_transformer import Block
from torch.utils.data import DataLoader
import random
import os
import json
from PIL import Image
import torchvision.transforms as transforms
from torchvision.ops import roi_align
from torch.distributions import Beta
import clip


# ===== 辅助函数和类 =====
def _convert_image_to_rgb(image):
    return image.convert("RGB")


def load_flickr_annotations(annotation_path):
    annotations = pd.read_table(
        annotation_path,
        sep='\t',
        header=None,
        names=['image_caption_id', 'caption']
    )
    annotations[['image_id', 'caption_id']] = annotations['image_caption_id'].str.split('#', expand=True)
    annotations['caption_id'] = annotations['caption_id'].astype(int)
    return annotations[['image_id', 'caption_id', 'caption']]


class Flickr30kDataset(torch.utils.data.Dataset):
    """Flickr30K数据集加载器（简化版）"""

    def __init__(self, image_dir, annotation_df, mode='train', num_crops=36,
                 rpn_proposals_file=None, rpn_ratio=0.5):
        self.image_dir = image_dir
        self.mode = mode
        self.num_crops = num_crops
        self.rpn_ratio = rpn_ratio
        self.annotation_df = annotation_df
        self.tokenizer = BertTokenizer.from_pretrained('./my_bert')

        # 加载RPN建议（如果有）
        self.rpn_proposals = None
        if rpn_proposals_file:
            with open(rpn_proposals_file, 'r') as f:
                self.rpn_proposals = json.load(f)

        # 图像和标注映射
        self.image_to_captions = annotation_df.groupby('image_id')['caption'].apply(list).to_dict()
        self.image_ids = list(self.image_to_captions.keys())
        self.image_paths = [os.path.join(image_dir, img_id) for img_id in self.image_ids]
        self.captions = [self.image_to_captions[img_id] for img_id in self.image_ids]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_id = self.image_ids[idx]

        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            # 全局图像
            w, h = image.size
            global_img = transform(image)

            crop_imgs = []

            # ====== 新增：面积过滤 ======
            # 计算原图总面积
            original_area = w * h
            # 筛选面积大于20%原图面积的提案框
            valid_proposals = []
            if self.rpn_proposals:
                # 获取RPN提案或使用后备方案
                proposals = self.rpn_proposals.get(image_id.split('.')[0], [])
                for box in proposals:
                    x1, y1, x2, y2 = box
                    box_w = x2-x1
                    box_h = y2 - y1
                    box_area = box_w * box_h
                    # 仅保留面积大于10%原图面积的提案框
                    if box_area >= 0.15 * original_area:
                        valid_proposals.append(box)
                # ====== 面积过滤结束 ======

            # 使用过滤后的提案框
            proposals = valid_proposals

            # 局部裁剪
            # ====== 新增：混合RPN建议框与随机裁剪 ======
            # 计算RPN建议框和随机裁剪的数量
            num_rpn = int(self.num_crops * self.rpn_ratio)  # RPN建议框数量
            num_random = self.num_crops - num_rpn  # 随机裁剪数量

            if self.rpn_proposals and len(proposals) > 0 and num_rpn > 0:
                # 确保不超过可用提案数量
                actual_rpn = min(num_rpn, len(proposals))
                # 随机选择提案
                indices = np.random.choice(len(proposals), actual_rpn, replace=False)
                for i in indices:
                    x1, y1, x2, y2 = proposals[i]
                    crop_img = image.crop((x1, y1, x2, y2))
                    crop_img = transform(crop_img)
                    crop_imgs.append(crop_img)
            else:
                actual_rpn = 0

            for _ in range(num_random + (num_rpn - actual_rpn)):  # 补足未采到的RPN数量
                crop_imgs.append(transform_random_crop(image))

            random.shuffle(crop_imgs)
            crop_imgs = torch.stack(crop_imgs)
            images = torch.cat([global_img.unsqueeze(0), crop_imgs], dim=0)

            # 处理多个文本
            captions = self.image_to_captions[image_id]
            if self.mode == 'train':  # 训练模式随机选1个字幕
                selected_idx = random.randint(0, len(captions) - 1)
                selected_captions = [captions[selected_idx]]
                caption_ids = [selected_idx]  # 新增：记录选中的caption索引
            else:  # 验证/测试模式使用全部字幕
                selected_captions = captions
                caption_ids = list(range(len(captions)))  # 新增：记录所有caption索引
            input_ids_list = []
            attention_mask_list = []
            for caption in selected_captions:
                text_input = self.tokenizer(
                    caption,
                    padding='max_length',
                    max_length=77,
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids_list.append(text_input['input_ids'])
                attention_mask_list.append(text_input['attention_mask'])

            # 堆叠多个文本 [5, 77]
            input_ids = torch.cat(input_ids_list, dim=0)
            attention_mask = torch.cat(attention_mask_list, dim=0)

            return {
                'images': images,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'image_id': image_id,  # 添加图像ID用于后续对齐
                'caption_ids': caption_ids  # 新增：返回caption_ids用于评估
            }

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return None


def transform(image):
    """全局图像变换"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])(image)


def transform_random_crop(image):
    """局部图像随机裁剪变换"""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 0.9)),
        transforms.RandomHorizontalFlip(),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])(image)
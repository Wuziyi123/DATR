import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from torchvision import transforms
import os
from PIL import Image
import torch
import clip
import torch.nn as nn
import pandas as pd
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer


_tokenizer = _Tokenizer()

# ====================== 数据加载与预处理 ======================
def load_flickr_annotations(annotation_path):
    """
    加载Flickr30k标注文件并转换为结构化DataFrame
    格式：image_id#caption_id\tcaption_text
    """
    # 读取原始标注文件
    annotations = pd.read_table(
        annotation_path,
        sep='\t',
        header=None,
        names=['image_caption_id', 'caption']
    )

    # 分离图片ID和字幕ID
    annotations[['image_id', 'caption_id']] = annotations['image_caption_id'].str.split('#', expand=True)
    annotations['caption_id'] = annotations['caption_id'].astype(int)

    # 清理并重组数据
    annotations = annotations[['image_id', 'caption_id', 'caption']]
    # return annotations.drop(columns=['image_caption_id'])
    return annotations


class Flickr30kDataset(Dataset):
    """Flickr30k数据集加载器（支持多模态训练）"""

    def __init__(self, image_dir, annotation_df, num_crops=5, transform=None):
        """
        参数:
        image_dir: 图片目录路径
        annotation_df: 通过load_flickr_annotations加载的标注DataFrame
        num_crops: 每张图生成的局部裁剪数量
        transform: 图像预处理变换
        """
        self.image_dir = image_dir
        self.annotations = annotation_df
        self.num_crops = num_crops
        self.transform = transform or self.default_transform()
        # 加载CLIP tokenizer
        self.tokenizer = clip.tokenize

        # 创建图片ID到所有描述的映射
        self.image_to_captions = annotation_df.groupby('image_id')['caption'].apply(list).to_dict()
        self.image_ids = list(self.image_to_captions.keys())

        # === 新增属性 ===
        self.image_paths = []  # 存储所有图像完整路径
        self.captions = []  # 存储所有描述（每张图5个描述）
        # 初始化图像路径和描述
        for img_id in self.image_ids:
            # 构建完整图像路径（添加.jpg扩展名）
            img_path = os.path.join(self.image_dir, f"{img_id}")
            self.image_paths.append(img_path)
            # 存储该图像对应的5个描述
            self.captions.append(self.image_to_captions[img_id])


    def default_transform(self):
        """CLIP标准预处理"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            )
        ])

    def random_crop(self, image):
        """生成随机裁剪区域（数据增强）"""
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.1, 0.9)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            )
        ])(image)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, image_id)

        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"无法加载图像: {img_path}")
            return None, None

        # 生成全局+局部图像组
        global_img = self.transform(image)
        crop_imgs = torch.stack([self.random_crop(image) for _ in range(self.num_crops)])
        images = torch.cat([global_img.unsqueeze(0), crop_imgs], dim=0)

        # 获取所有描述
        captions = self.image_to_captions[image_id]

        # 文本描述编码
        tokenized_captions = torch.stack([
            self.tokenizer(caption, truncate=True)[0]
            for caption in captions
        ])

        return images, tokenized_captions


# ====================== 使用示例 ======================
if __name__ == "__main__":
    # 1. 加载标注数据
    annotation_path = '/home/wuziyi/Project/Flickr/Flickr30K_Entities/data/results_20130124.token'  # 替换为实际路径
    annotations_df = load_flickr_annotations(annotation_path)

    print("标注数据统计:")
    print(f"总图片数: {annotations_df['image_id'].nunique()}")
    print(f"总描述数: {len(annotations_df)}")
    print(f"每张图片平均描述数: {len(annotations_df) / annotations_df['image_id'].nunique():.2f}")

    # 2. 创建数据集
    image_dir = '/home/wuziyi/Project/Flickr/Flickr30K_Entities/data/flickr30k-images'  # 替换为实际路径
    dataset = Flickr30kDataset(
        image_dir=image_dir,
        annotation_df=annotations_df,
        num_crops=5
    )


    def collate_fn(batch):
        # 过滤无效样本
        batch = [b for b in batch if b[0] is not None]
        # 处理图像数据 [B, 6, 3, 224, 224]
        images = torch.stack([item[0] for item in batch])
        # 处理文本数据 [B, 5, 77] -> [B*5, 77]
        texts = torch.cat([item[1] for item in batch], dim=0)
        return images, texts

    # 3. 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 4. 测试数据加载
    sample_images, sample_captions = next(iter(dataloader))
    print("\n批量数据示例:")
    print(f"图像张量尺寸: {sample_images.shape}")  # [batch, 6, 3, 224, 224]
    print(f"描述token尺寸: {sample_captions.shape}")  # [batch, 77]
    # cv = sample_captions[0].tolist()
    print(f"描述示例: {_tokenizer.decode(sample_captions[0].tolist())}")

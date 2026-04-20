import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from torchvision import transforms
import os
from PIL import Image
import torch
import clip
import random
import pandas as pd
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


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


class Flickr30kDataset(Dataset):
    def __init__(self, image_dir, annotation_df, num_crops=5,
                 transform=None, rpn_proposals_file=None, rpn_ratio=0.6):
        self.image_dir = image_dir
        self.annotations = annotation_df
        self.num_crops = num_crops
        self.transform = transform or self.default_transform()
        self.tokenizer = clip.tokenize
        self.rpn_ratio = rpn_ratio  # RPN建议框的采样比例

        # 加载RPN提案
        if rpn_proposals_file and os.path.exists(rpn_proposals_file):
            with open(rpn_proposals_file, 'r') as f:
                self.rpn_proposals = json.load(f)
            print(f"Loaded RPN proposals from {rpn_proposals_file}")
        else:
            self.rpn_proposals = {}
            print("No RPN proposals found, using random crops")

        # 图像和标注映射
        self.image_to_captions = annotation_df.groupby('image_id')['caption'].apply(list).to_dict()
        self.image_ids = list(self.image_to_captions.keys())
        self.image_paths = [os.path.join(image_dir, img_id) for img_id in self.image_ids]
        self.captions = [self.image_to_captions[img_id] for img_id in self.image_ids]

        # 预处理变换
        self.crop_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            )
        ])

        # 后备随机裁剪
        self.random_crop = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 0.9)),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            )
        ])

    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            )
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"无法加载图像: {img_path}")
            return None, None

        w, h = image.size
        global_img = self.transform(image)

        # 获取RPN提案或使用后备方案
        proposals = self.rpn_proposals.get(image_id.split('.')[0], [])
        crop_imgs = []

        # ====== 新增：面积过滤 ======
        # 计算原图总面积
        original_area = w * h
        # 筛选面积大于20%原图面积的提案框
        valid_proposals = []
        for box in proposals:
            x1, y1, x2, y2 = box
            box_w = max(0, min(x2, w) - max(0, x1))  # 确保宽度有效
            box_h = max(0, min(y2, h) - max(0, y1))  # 确保高度有效
            box_area = box_w * box_h

            # 仅保留面积大于10%原图面积的提案框
            if box_area >= 0.15 * original_area:
                valid_proposals.append(box)
        # ====== 面积过滤结束 ======

        # 使用过滤后的提案框
        proposals = valid_proposals
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
                # 确保坐标有效
                # x1 = max(0, min(int(x1), w - 1))
                # y1 = max(0, min(int(y1), h - 1))
                # x2 = max(x1 + 1, min(int(x2), w))
                # y2 = max(y1 + 1, min(int(y2), h))
                crop_img = image.crop((x1, y1, x2, y2))
                crop_img = self.crop_transform(crop_img)
                crop_imgs.append(crop_img)
        else:
            actual_rpn = 0

        for _ in range(num_random + (num_rpn - actual_rpn)):  # 补足未采到的RPN数量
            crop_imgs.append(self.random_crop(image))

        random.shuffle(crop_imgs)

        crop_imgs = torch.stack(crop_imgs)
        images = torch.cat([global_img.unsqueeze(0), crop_imgs], dim=0)

        # 文本处理
        captions = self.image_to_captions[image_id]
        tokenized_captions = torch.stack([
            self.tokenizer(caption, truncate=True)[0]
            for caption in captions
        ])

        return images, tokenized_captions


def _convert_image_to_rgb(image):
    return image.convert("RGB")


# ====================== 主函数 ======================
if __name__ == "__main__":
    # 1. 配置路径
    annotation_path = '/home/wuziyi/Project/Flickr/Flickr30K_Entities/data/results_20130124.token'
    image_dir = '/home/wuziyi/Project/Flickr/Flickr30K_Entities/data/flickr30k-images'
    proposal_file = '/home/wuziyi/Project/Flickr/Flickr30K_Entities/flickr30k_rpn_proposals-U.json'  # 预生成的提案文件

    # 2. 加载标注
    annotations_df = load_flickr_annotations(annotation_path)

    # 3. 创建数据集
    dataset = Flickr30kDataset(
        image_dir=image_dir,
        annotation_df=annotations_df,
        num_crops=5,
        rpn_proposals_file=proposal_file,
        rpn_ratio=0.6,
    )


    # 4. 数据加载器
    def collate_fn(batch):
        batch = [b for b in batch if b[0] is not None]
        images = torch.stack([item[0] for item in batch])
        texts = torch.cat([item[1] for item in batch], dim=0)
        return images, texts


    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,  # 多进程安全
        collate_fn=collate_fn
    )

    # 6. 测试
    sample_images, sample_captions = next(iter(dataloader))
    print("\n批量数据示例:")
    print(f"图像张量尺寸: {sample_images.shape}")
    print(f"描述token尺寸: {sample_captions.shape}")
    print(f"描述示例: {_tokenizer.decode(sample_captions[0].tolist())}")
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from torchvision import transforms, ops
import os
from PIL import Image
import torch
import clip
import torch.nn as nn
import pandas as pd
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms.functional as F

_tokenizer = _Tokenizer()


# ====================== 预生成RPN提案 ======================
def generate_rpn_proposals(image_dir, output_file, num_proposals=100):
    """
    预生成RPN提案并保存到JSON文件
    避免数据加载时重复初始化模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型（仅需运行一次）
    backbone = torchvision.models.resnet50(pretrained=True)
    backbone.out_channels = 2048
    anchor_generator = AnchorGenerator()
    model = FasterRCNN(
        backbone,
        num_classes=2,  # 背景+前景
        rpn_anchor_generator=anchor_generator
    ).to(device).eval()

    proposals_dict = {}
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        image_id = os.path.splitext(img_file)[0]

        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

            with torch.no_grad():
                # 仅运行RPN部分
                features = model.backbone(img_tensor)
                proposals, _ = model.rpn(img_tensor, features)
                dense_proposals = proposals[0].to_dense()  # 稀疏→密集转换

                # 选择top-k提案
                scores = dense_proposals[0][:, 4]  # 置信度分数
                top_indices = scores.topk(min(num_proposals, len(scores))).indices
                top_proposals = dense_proposals[top_indices, :4].cpu().numpy().tolist()

                proposals_dict[image_id] = top_proposals

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            proposals_dict[image_id] = []

    # 保存到文件
    with open(output_file, 'w') as f:
        json.dump(proposals_dict, f)

    return proposals_dict


# ====================== 数据加载与预处理 ======================
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


class Flickr30kDataset(Dataset):
    def __init__(self, image_dir, annotation_df, num_crops=5,
                 transform=None, proposal_file=None):
        self.image_dir = image_dir
        self.annotations = annotation_df
        self.num_crops = num_crops
        self.transform = transform or self.default_transform()
        self.tokenizer = clip.tokenize

        # 加载RPN提案
        self.proposals = {}
        if proposal_file and os.path.exists(proposal_file):
            try:
                with open(proposal_file, 'r') as f:
                    self.proposals = json.load(f)
                print(f"Loaded proposals from {proposal_file}")
            except:
                print(f"Failed to load proposals from {proposal_file}")

        # 图像和标注映射
        self.image_to_captions = annotation_df.groupby('image_id')['caption'].apply(list).to_dict()
        self.image_ids = list(self.image_to_captions.keys())
        self.image_paths = [os.path.join(image_dir, img_id) for img_id in self.image_ids]
        self.captions = [self.image_to_captions[img_id] for img_id in self.image_ids]

        # 变换初始化
        self.crop_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)
            )
        ])

        # 后备随机裁剪
        self.fallback_crop = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 0.9)),
            transforms.CenterCrop(224),
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

        # 全局图像
        global_img = self.transform(image)

        # 局部图像生成
        crop_imgs = []
        img_proposals = self.proposals.get(image_id, [])

        if len(img_proposals) >= self.num_crops:
            # 从RPN提案中随机选择
            indices = np.random.choice(len(img_proposals), self.num_crops, replace=False)
            for idx in indices:
                x1, y1, x2, y2 = img_proposals[idx]
                w, h = image.size

                # 确保坐标有效
                x1 = max(0, min(int(x1), w - 1))
                y1 = max(0, min(int(y1), h - 1))
                x2 = max(x1 + 1, min(int(x2), w))
                y2 = max(y1 + 1, min(int(y2), h))

                # 裁剪并转换
                crop_img = image.crop((x1, y1, x2, y2))
                crop_img = self.crop_transform(crop_img)
                crop_imgs.append(crop_img)
        else:
            # 后备：随机裁剪
            for _ in range(self.num_crops):
                crop_imgs.append(self.fallback_crop(image))

        crop_imgs = torch.stack(crop_imgs)
        images = torch.cat([global_img.unsqueeze(0), crop_imgs], dim=0)

        # 文本处理
        captions = self.image_to_captions[image_id]
        tokenized_captions = torch.stack([
            self.tokenizer(caption, truncate=True)[0]
            for caption in captions
        ])

        return images, tokenized_captions


# ====================== 主函数 ======================
if __name__ == "__main__":
    # 1. 配置路径
    annotation_path = '/home/wuziyi/Project/Flickr/Flickr30K_Entities/data/results_20130124.token'
    image_dir = '/home/wuziyi/Project/Flickr/Flickr30K_Entities/data/flickr30k-images'
    proposal_file = '/home/wuziyi/Project/Flickr/Flickr30K_Entities/flickr30k_proposals.json'  # 预生成的提案文件

    # 2. 预生成RPN提案（只需运行一次）
    if not os.path.exists(proposal_file):
        print("Generating RPN proposals...")
        generate_rpn_proposals(image_dir, proposal_file)

    # 3. 加载标注
    annotations_df = load_flickr_annotations(annotation_path)

    # 4. 创建数据集
    dataset = Flickr30kDataset(
        image_dir=image_dir,
        annotation_df=annotations_df,
        num_crops=5,
        proposal_file=proposal_file
    )


    # 5. 数据加载器
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
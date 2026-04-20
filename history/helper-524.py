import gc
import os
import random

import cv2
import numpy as np
import torch
import json
import pickle
import matplotlib.pyplot as plt

from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from clip import clip
from torchvision.datasets import ImageNet, ImageFolder, Places365
from my_datasets import *
from utils import (
    openai_imagenet_classes,
    imagenet_classes,
    imagenet_a_lt,
    imagenet_r_lt,
)
from sam_sample import get_crop_Images
from edge_sam import SamPredictor, sam_model_registry
import torch.nn as nn
from timm.models.vision_transformer import Block
from functools import partial
from edge_sam.utils.transforms import ResizeLongestSide
from skimage.transform import resize


def load_json(filename):
    if not filename.endswith(".json"):
        filename += ".json"
    with open(filename, "r") as fp:
        return json.load(fp)


def set_seed(seed):
    print(f"Setting seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(data_path, dataset_name, custom_loader):
    data_path = data_path

    if dataset_name == MyDataset.ImageNet:
        dataset = ImageNet(
            data_path,
            split="val",
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.ImageNetV2:
        dataset = ImageNetV2Dataset(
            location=data_path,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.ImageNetR:
        dataset = ImageFolder(
            root=data_path,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.ImageNetS:
        dataset = ImageFolder(
            root=data_path,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.ImageNetA:
        dataset = ImageFolder(
            root=data_path,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.CUB:
        dataset = CUBDataset(
            data_path,
            train=False,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.Food101:
        dataset = Food101(
            data_path,
            transform=None,
            loader=custom_loader,
            split="test",
            download=False,
        )

    elif dataset_name == MyDataset.OxfordIIITPet:
        dataset = OxfordIIITPet(
            data_path,
            transform=None,
            split="test",
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.Place365:
        dataset = Places365(
            data_path,
            transform=None,
            loader=custom_loader,
            download=False,
            split="val",
            small=False,
        )

    elif dataset_name == MyDataset.DTD:
        dataset = DTD(
            data_path,
            # transform=None,
            loader=custom_loader,
            split="test",
            download=False,
        )

    return dataset


def wordify(string):
    word = string.replace("_", " ")
    return word


def load_classes(dataset_name):
    with open(
        f"features/{dataset_name}/{dataset_name}.json",
        "r",
    ) as f:
        classes = json.load(f)

    wordify_classes = []
    for c in classes:
        wordify_classes.append(wordify(c))

    return wordify_classes


def generate_weights(
    method,
    model,
    dataset_name,
    tt_scale=None,
    device=None,
):
    templates = None
    make_sentence = False
    is_template = True

    # if dataset start with imagenet
    if dataset_name.startswith(MyDataset.ImageNet):
        classes = (
            openai_imagenet_classes
            if method in ["clip-d", "waffle"]
            else imagenet_classes
        )
    else:
        classes = load_classes(dataset_name)

    print(f"Creating {method} text embeddings...")

    if method != "clip":
        if method == "ours":
            load_file = "cupl"
        elif method == "cupl":
            load_file = "cupl"
        elif method == "waffle":
            load_file = "clip-d"
        else:
            load_file = method

        with open(f"prompts/{dataset_name}/{load_file}.json") as f:
            templates = json.load(f)

        if method in ["waffle", "clip-d", "cupl", "ours"]:
            is_template = False

        if method == "clip-d":
            make_sentence = True

        if method == "waffle":
            templates = construct_random(templates)

    zeroshot_weights = zeroshot_classifier(
        model,
        classes,
        templates,
        is_template,
        make_sentence,
        tt_scale,
        device,
    )

    return zeroshot_weights


class SAMEnhancedCLIP(torch.nn.Module):
    def __init__(self, clip_model, device):
        super().__init__()
        self.device = device
        self.patch_size = 16
        sam = sam_model_registry["edge_sam"](checkpoint="edge_sam_3x.pth")
        sam.half().to(device).eval()
        self.predictor = SamPredictor(sam)
        self.clip = clip_model
        self.num_heads = 12
        # 注册hook处理中间特征
        self.attention_masks = None
        # self.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, outputs):
        """在每层Transformer前注入掩码"""
        if self.attention_masks is not None:
            # for layer in self.vit.encoder.layers:
            #     layer.self_attn.attention_mask = self.attention_masks
            for layer in self.clip.visual.transformer.resblocks:
                layer.attn_mask = self.attention_masks


    def _register_attention_hooks(self):
        """在每层Transformer前注入掩码"""
        if self.attention_masks is not None:
            for layer in self.clip.visual.transformer.resblocks:
                layer.attn_mask = self.attention_masks
        else:
            for layer in self.clip.visual.transformer.resblocks:
                layer.attn_mask = None

    def _prepare_mask(self, binary_mask):
        """将原始二值掩码转换为注意力引导矩阵"""
        # Step 1: 下采样到patch网格
        # grid_size = 224 // self.patch_size
        mask_patches = F.avg_pool2d(
            binary_mask.float(),
            kernel_size=self.patch_size,
            stride=self.patch_size
        ).squeeze()  # [14,14]

        # Step 2: 构建序列掩码
        batch_size = mask_patches.size(0)
        cls_tokens = torch.ones(batch_size, 1, device=self.device)  # 批量CLS token
        seq_mask = torch.cat([
            cls_tokens,
            mask_patches.flatten(start_dim=1)
        ], dim=1)  # [197]

        # Step 3: 生成注意力引导矩阵 (优化广播机制)
        attn_guidance = torch.einsum(
            'bi,bj->bij',
            seq_mask,
            seq_mask
        ) * 3.0  # 外积运算 [B, 197, 197]

        # Step 4: 扩展到多头 [B, num_heads, 197, 197]
        attn_guidance = attn_guidance.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        # 合并批次和头维度
        attn_mask_3d = attn_guidance.reshape(-1, attn_guidance.size(2), attn_guidance.size(3))
        return attn_mask_3d, seq_mask

    def visualize_tensors(self, rgb_tensor, gray_tensor):
        # 处理RGB图像
        rgb_image = rgb_tensor.to(torch.float32).cpu().numpy()
        # 调整维度顺序：PyTorch(CHW) -> Matplotlib(HWC)
        rgb_image = np.transpose(rgb_image, (1, 2, 0))

        # 处理灰度图像
        gray_image = gray_tensor.to(torch.float32).cpu().numpy()
        # 扩展维度：HxW -> HxWx1
        gray_image = np.expand_dims(gray_image, axis=-1)

        # 创建画布
        plt.figure(figsize=(12, 6))

        # 显示RGB图像
        plt.subplot(1, 2, 1)
        plt.imshow(rgb_image)
        plt.title('RGB Image (3x224x224)')
        plt.axis('off')

        # 显示灰度图像
        plt.subplot(1, 2, 2)
        plt.imshow(gray_image, cmap='gray')
        plt.title('Grayscale Feature Map (14x14)')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def forward(self, x, sampling_points, paths):
        # 重置注意力为空
        self.attention_masks = None
        self._register_attention_hooks()
        # 生成掩码 304 * 512 == 16 * (1+18) * 512
        with torch.no_grad():
            B, _, C, H, W = x.shape  # 16 19 3 224 224
            batch_masks = []
            image_global = x[:, 0, :, :, :].half()
            image_crop = x[:, 1:, :, :, :].reshape(-1, C, H, W).contiguous().half()
            outputs_crop = self.clip.encode_image(image_crop)

            for i in range(len(paths)):
                image = cv2.imread(paths[i])
                image_plt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.predictor.set_image(image_plt)

                with torch.cuda.amp.autocast():
                    masks, scores, logits = self.predictor.predict(
                        point_coords=np.array(sampling_points[i]),
                        point_labels=np.ones(len(sampling_points[i]), dtype=np.int64),
                        num_multimask_outputs=1,
                        use_stability_score=True
                    )
                mask_tensor = torch.from_numpy(masks[0].astype(np.float32))
                mask_4d = mask_tensor.unsqueeze(0).unsqueeze(0)
                resized = F.interpolate(mask_4d, size=(224, 224),mode='nearest').squeeze().numpy()  # 移除通道维度 → [B,H,W]

                # 确保严格二值化
                binary_mask = (resized >= 0.5).astype(np.float16)
                batch_masks.append(binary_mask)

            tensors = [torch.from_numpy(arr) for arr in batch_masks]
            mask_imgs = torch.stack(tensors).to(self.device)

            # 生成注意力引导矩阵
            self.attention_masks, seq_mask = self._prepare_mask(mask_imgs)
            # 原始ViT处理流程
            self._register_attention_hooks()

            rgb_tensor = image_global[0,:,:,:]      # 3,14,14
            gray_tensor = seq_mask[0, 1:].reshape(14, 14)  # 197
            # 执行可视化
            self.visualize_tensors(rgb_tensor, gray_tensor)

            outputs_global = self.clip.encode_image(image_global)

            # 计算插入后的总长度
            total_length = outputs_global.shape[0] + outputs_crop.shape[0]
            result = torch.zeros(total_length, outputs_global.shape[1], dtype=outputs_crop.dtype, device=self.device)  # (304, 512)
            index = total_length // outputs_global.shape[0]
            # 生成插入位置的索引（每隔18个位置插入1个）
            insert_positions = torch.arange(0, total_length, index, device=self.device)  # [0, 19, 38, 57, ..., 285]
            # 生成非插入位置的索引（即 outputs_crop 应该填充的位置）
            non_insert_mask = torch.ones(total_length, dtype=torch.bool, device=self.device)  # 初始全 True
            non_insert_mask[insert_positions] = False  # 插入位置设为 False
            non_insert_indices = torch.where(non_insert_mask)[0]  # 获取非插入位置的索引

            # 填充 outputs_crop 到非插入位置
            result[non_insert_indices] = outputs_crop
            # 填充 outputs_global 到插入位置
            result[insert_positions] = outputs_global

        return result


class GlobalExpert(nn.Module):
    """全局语义专家（增大感受野版本）"""

    def __init__(self, dim, expansion_ratio=4, num_heads=12):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)

        # 扩展FFN维度
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion_ratio),
            nn.GELU(),
            nn.Linear(dim * expansion_ratio, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # 全局注意力增强
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return x


class LocalExpert(nn.Module):
    """局部细节专家（轻量化设计）"""

    def __init__(self, dim, patch_size=16):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=3,
                                 padding=1, groups=dim)  # Depthwise卷积

        self.channel_mixer = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim)
        )

    def forward(self, x):
        x = x.permute(1, 0, 2)
        B, N, C = x.shape  # 400 196 768
        H = W = int(N ** 0.5)  # 400, 196, 768
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.dw_conv(x)  # 局部特征提取
        x = x.permute(0, 2, 3, 1).view(B, -1, C)
        return self.channel_mixer(x).permute(1, 0, 2)


class MoELayer(nn.Module):
    """混合专家层（包含路由机制）"""

    def __init__(self, dim, num_local_experts=4, in_features=None,
                hidden_features=None,
                act_layer=None, bias=None, drop=None,
                **kwargs):
        super().__init__()
        self.global_expert = GlobalExpert(dim)
        self.local_experts = nn.ModuleList(
            [LocalExpert(dim) for _ in range(num_local_experts)]
        )

        # 路由网络
        self.gate_global = nn.Linear(dim, 1)
        self.gate_local = nn.Linear(dim, num_local_experts)

        # 噪声注入
        self.noise_mean = 0.0
        self.noise_std = 0.1

    def add_noise(self, logits):
        noise = torch.randn_like(logits) * self.noise_std + self.noise_mean
        return logits + noise

    def forward(self, x):
        # 全局路径
        # cls_token = x[:, 0:1]  (1, 400, 768)
        cls_token = x[0:1, :]
        global_weights = torch.sigmoid(self.add_noise(
            self.gate_global(cls_token)
        )).squeeze(-1)

        # 局部路径  (196, 400, 768) (400, 768) local_weights = (400, 4)
        patch_tokens = x[1:, :]
        local_weights = torch.softmax(self.add_noise(
            self.gate_local(patch_tokens.mean(0))
        ), dim=-1)

        # 专家计算 (1, 400, 768)
        global_out = self.global_expert(cls_token)

        # 动态容量分配
        local_outs = []
        for i in range(local_weights.shape[-1]):
            mask = (local_weights.argmax(-1) == i)
            if mask.sum() > 0:
                selected = patch_tokens[:, mask, :]
                local_out = self.local_experts[i](selected)
                temp = local_weights[mask, i].view(1, -1, 1).expand_as(local_out)
                local_outs.append(local_out * temp)

        # 特征融合
        tensor_local = torch.cat(local_outs, dim=1)
        tensor_global = global_out * global_weights.unsqueeze(-1)
        combined = torch.cat([tensor_local, tensor_global], dim=0)
        # combined = torch.cat([global_out * global_weights.unsqueeze(-1)] + local_outs, dim=0)
        return combined


class GOAL_CLIP(nn.Module):
    def __init__(self, clip_model, num_local_experts=4):
        super().__init__()
        # 初始化CLIP骨干
        self.visual = clip_model.visual
        self.text_model = clip_model.transformer

        # 替换ViT中的MLP层为MoE
        for i in range(len(self.visual.transformer.resblocks)):
            original_block = self.visual.transformer.resblocks[i]
            dim = original_block.mlp[0].in_features
            new_block = Block(
                dim, num_heads=original_block.attn.num_heads,
                mlp_layer = partial(MoELayer, dim=dim, num_local_experts=num_local_experts),
                qkv_bias=True
            ).half()
            self.visual.transformer.resblocks[i] = new_block

        # SAM初始化（局部区域提取）
        self.sam = sam_model_registry["edge_sam"](checkpoint="edge_sam_3x.pth")

        # 对齐投影层
        self.global_proj = nn.Linear(512, 256)
        self.local_proj = nn.ModuleList([
            nn.Linear(512, 256) for _ in range(num_local_experts)
        ])

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype


    def forward(self, images, texts):
        # 图像特征提取
        global_feat = self.visual(images.type(self.dtype))
        # 文本特征提取（支持长文本）
        # text_feat = self.process_long_text(texts)
        text_feat = None

        # 局部区域检测
        local_regions = self.extract_local_regions(images)
        local_feats = [self.visual(region.type(self.dtype)) for region in local_regions]

        # 多专家特征融合
        combined_feat = self.fuse_features(global_feat, local_feats)

        # 对比学习
        logits = combined_feat @ text_feat.T
        return logits

    def process_long_text(self, texts):
        # 长文本处理（位置插值）
        seq_len = texts.shape[1]
        if seq_len > 77:
            pos_emb = self.text_model.positional_embedding
            new_pos_emb = nn.functional.interpolate(
                pos_emb.unsqueeze(0), size=seq_len, mode='linear')
            self.text_model.positional_embedding = new_pos_emb.squeeze(0)
        return self.text_model(texts)

    def generate_random_boxes(self, batch_size, img_size=224, num_boxes=4):
        """生成随机边界框[6](@ref)"""
        boxes = []
        for _ in range(batch_size):
            img_boxes = []
            for _ in range(num_boxes):
                # 随机生成框尺寸（最小边长20px）
                w = np.random.randint(100, img_size // 2)
                h = np.random.randint(100, img_size // 2)
                # 随机生成起始点
                x1 = np.random.randint(0, img_size - w)
                y1 = np.random.randint(0, img_size - h)
                x2, y2 = x1 + w, y1 + h
                img_boxes.append([x1, y1, x2, y2])
            boxes.append(torch.tensor(img_boxes))
        return torch.stack(boxes)

    def prepare_batch(self, images, sam_device):
        """将4D张量转换为Edge-SAM的批处理格式"""
        # 添加归一化 [2](@ref)
        normalized_tensor = (images * 255).to(torch.uint8)
        normalized_tensor = normalized_tensor[:, [2, 1, 0], :, :]  # RGB转BGR
        resize_transform = ResizeLongestSide(224)

        # 生成随机边界框
        random_boxes = self.generate_random_boxes(images.size(0))

        batched_input = []
        original_size = images.shape[2:]
        # 遍历批次中的每个图像 [网页2][网页4]
        for i in range(images.size(0)):
            # 转换边界框坐标
            adjusted_boxes = resize_transform.apply_boxes_torch(
                random_boxes[i].to(sam_device),
                original_size=(224, 224)
            )
            # 构造输入字典
            batched_input.append({
                'image': normalized_tensor[i],
                'boxes': adjusted_boxes,
                'original_size': original_size,  # 原始H,W
            })
        return batched_input

    def extract_local_regions(self, images, visualize=True):
        """主处理流程"""
        batched_input = self.prepare_batch(images, self.sam.device)

        # SAM推理 [网页5]
        with torch.no_grad():
            features = self.sam(batched_input, num_multimask_outputs=1)
            masks = [f['masks'] for f in features]  # 提取各样本的掩码
            boxes = batched_input[0]['boxes'].cpu().numpy()
            # 新增可视化逻辑
            if visualize and len(masks) > 0:
                self._visualize_third_sample(images[0], masks[0], boxes)
            return self.generate_boxes(masks)

    def _visualize_third_sample(self, original, masks, boxes):
        # 二值化掩码并还原原始尺寸
        masks = (masks.squeeze(1) > 0.5).cpu().numpy()  # [B,1,H,W] -> [B,H,W]
        original = original.clone().permute(1, 2, 0).contiguous()

        plt.figure(figsize=(15, 5))

        # 原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(original.cpu().numpy())
        plt.title("Original Image")

        # 边界框叠加
        plt.subplot(1, 3, 2)
        box_img = original.cpu().numpy()
        box_img = np.ascontiguousarray(box_img[:, :, ::-1])
        for box in boxes:
            cv2.rectangle(box_img,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 255, 0), 2)
        plt.imshow(box_img)
        plt.title("Bounding Boxes")

        # 掩码叠加
        plt.subplot(1, 3, 3)
        mask_img = original.cpu().numpy().copy()
        for mask in masks:
            mask_img[mask] = [255, 0, 0]  # 红色覆盖
        plt.imshow(mask_img)
        plt.title("Mask Overlay")

        plt.show()
        pass

    def generate_boxes(self, masks, score_threshold=0.5):
        """从掩码生成边界框 [网页7][网页8]"""
        all_boxes = []

        for batch_idx, mask_batch in enumerate(masks):
            # mask_batch形状: (num_masks, 1, H, W)
            batch_boxes = []

            # 遍历每个预测掩码 [网页9]
            for mask in mask_batch.squeeze(1):  # 去除通道维度
                # 过滤低置信度区域 [网页5]
                if mask.max() < score_threshold:
                    continue

                # 获取有效区域坐标 [网页10]
                y, x = torch.where(mask > score_threshold)
                if len(y) == 0 or len(x) == 0:
                    continue

                # 计算边界框 [网页9]
                x_min = x.min().item()
                x_max = x.max().item()
                y_min = y.min().item()
                y_max = y.max().item()
                batch_boxes.append([x_min, y_min, x_max, y_max])

            # 转换为Tensor [网页4]
            if batch_boxes:
                boxes_tensor = torch.tensor(batch_boxes,
                                            device=self.device,
                                            dtype=torch.float32)
                # 可选NMS（参考网页2）
                all_boxes.append(boxes_tensor)
            else:
                all_boxes.append(torch.zeros((0, 4), device=self.device))

        return all_boxes

    # def extract_local_regions(self, images):
    #     # 使用SAM提取局部区域
    #     with torch.no_grad():
    #         features = self.sam(images)
    #         masks = features['masks']
    #         return self.generate_boxes(masks)

    def fuse_features(self, global_feat, local_feats):
        # 动态门控融合
        global_emb = self.global_proj(global_feat)
        local_embs = [proj(feat) for proj, feat in zip(self.local_proj, local_feats)]
        weights = torch.softmax(torch.cat([global_emb.mean(-1)] +
                                          [emb.mean(-1) for emb in local_embs]), dim=-1)
        return weights[0] * global_emb + sum(w * emb for w, emb in zip(weights[1:], local_embs))


class DynamicNoiseFilter(nn.Module):
    """基于不确定性的动态噪声过滤模块"""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.uncertainty_proj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )
        self.semantic_router = nn.MultiheadAttention(dim, num_heads)

    def forward(self, text_emb, image_emb):
        # 语义置信度计算 [7,13](@ref)
        uncertainty = torch.sigmoid(self.uncertainty_proj(text_emb))  # (B,L,1)

        # 跨模态语义路由
        refined_emb, _ = self.semantic_router(
            text_emb, image_emb, image_emb,
            key_padding_mask=(uncertainty < 0.5).squeeze()
        )
        return refined_emb * uncertainty


class MemoryBank(nn.Module):
    """基于局部语义的跨样本缓存池"""

    def __init__(self, feat_dim=256, capacity=1024):
        super().__init__()
        self.register_buffer('feat_bank', torch.randn(capacity, feat_dim))
        self.register_buffer('label_bank', torch.randint(0, 1000, (capacity,)))
        self.ptr = 0

    def update(self, features, labels):
        batch_size = features.size(0)
        if self.ptr + batch_size > self.capacity:
            self.ptr = 0
        self.feat_bank[self.ptr: self.ptrptr+batch_size] = features.detach()
        self.label_bank[self.ptr: self.ptrptr+batch_size] = labels.detach()
        self.ptr += batch_size

    def retrieve(self, query, k=5):
        # 语义相似度检索 [5](@ref)
        sim = F.cosine_similarity(query.unsqueeze(1), self.feat_bank.unsqueeze(0), dim=-1)
        topk_idx = sim.topk(k, dim=1).indices
        return self.feat_bank[topk_idx]


class UncertaintyOrthoFusion(nn.Module):
    """基于置信度的正交特征融合"""
    def __init__(self, dim):
        super().__init__()
        self.ortho_proj = nn.Linear(dim, dim)

    def forward(self, global_feat, local_feats, confidences):
        # 正交投影去冗余 [14](@ref)
        proj_weights = torch.mm(
            global_feat,
            torch.stack(local_feats).permute(1, 0)
        ).softmax(dim=-1)

        # 加权正交融合
        ortho_feats = []
        for feat, weight in zip(local_feats, proj_weights):
            ortho = feat - self.ortho_proj(global_feat) * weight
            ortho_feats.append(ortho * confidences)
        return torch.cat([global_feat] + ortho_feats, dim=-1)


def load_precomputed_features(
    model,
    dataset_name: str,
    model_size: str,
    alpha: float,
    n_samples: int,
    batch_size: int,
    num_workers: int,
    data_path: str,
    custom_loader: callable,
    device: torch.device,
    processor
):
    save_file = (dataset_name + "-" + model_size).replace("/", "-")
    save_root = f"features/{dataset_name}"

    # if save_root not exist, create it
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    filename = os.path.join(save_root, f"{save_file}-{alpha}-{n_samples}.pkl")

    if os.path.exists(filename):
        print(f"Loading {filename}...")
        load_res = pickle.load(open(filename, "rb"))
    else:
        print(f"File {filename} not found, precomputing features...")
        dataset = load_dataset(
            data_path=data_path,
            dataset_name=dataset_name,
            custom_loader=custom_loader,
        )

        dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # CUB classes need to be manually processed
        # if hasattr(dataset, "classes") and dataset_name != MyDataset.CUB:
        #     classes = dataset.classes
        #     classes_file = os.path.join(save_root, f"{dataset_name}.json")
        #     if not os.path.exists(classes_file):
        #         with open(classes_file, "w") as f:
        #             json.dump(classes, f)

        precomputed_features = []
        image_features_tensor = []
        target = []
        sam_enhanced_clip = SAMEnhancedCLIP(model, device)

        with torch.no_grad():
            for batch in tqdm(dataloader):
                # images = B=12, NS=1, C=3, H=224, W=224
                images, labels, paths = batch
                labels = labels.to(device)
                images = images.to(device)

                b, ns = images.shape[:2]
                images = images.flatten(0, 1)
                # combined_images = combined_images.flatten(0, 1)
                # images = images.view(-1, *images.shape[2:])

                # 创建GOAL_CLIP模型
                model = GOAL_CLIP(
                    clip_model=model,
                    num_local_experts=4
                ).to(device)

                logits = model(images, labels)
                print(f"输出logits形状: {logits.shape}")  # 应为[batch_size, batch_size]
                print(f"样本预测值:\n{logits}")


                # 掩码增强
                batch_selected_points = get_crop_Images(paths)
                image_features = sam_enhanced_clip(images, batch_selected_points, paths)
                # image_features = model.encode_image(images)
                del images

                image_features = F.normalize(image_features)
                image_features = image_features.view(b, ns, -1).contiguous() # b,ns,d

                # 优化: 分离计算图并转移数据到CPU
                patch_features = image_features[:, 1:].detach()  # 分离计算图
                image_features_main = image_features[:, :1].detach()
                del image_features  # 及时释放

                weight_image = (image_features_main * patch_features).sum(
                    dim=-1, keepdim=True
                )

                # 负载均衡损失计算
                # expert_weights = ...  # 从MoE层获取专家激活权重（形状：[batch_size, num_experts]）
                # balance_loss = torch.mean(torch.stack([w.pow(2).mean() for w in expert_weights]))
                # return contrastive_loss + self.lambda_balance * balance_loss

                # DS = (image_features * patch_features)  # 4,60,512 = (4,1,512) * （4,60,512）
                # weight_image = (batch, nsamples, 1)
                patch_with_weights = torch.cat([patch_features, weight_image], -1)

                # 优化: 数据存储到CPU
                precomputed_features.append(patch_with_weights.cpu())
                target.append(labels.cpu())
                image_features_tensor.append(image_features_main.squeeze(1).cpu())
                # 优化: 强制垃圾回收显存
                del patch_features, image_features_main, weight_image, patch_with_weights
                torch.cuda.empty_cache()

                # precomputed_features.append(patch_with_weights)
                # target.append(labels)
                # image_features_tensor.append(image_features.squeeze(1))

        # 最终合并时转换回原精度（如需）
        load_res = {
            "patches": torch.cat([x for x in precomputed_features], dim=0),
            "images": torch.cat(image_features_tensor, dim=0),
            "labels": torch.cat(target, dim=0),
        }
        # load_res = {
        #     "patches": torch.cat(precomputed_features, dim=0),
        #     "images": torch.cat(image_features_tensor, dim=0),
        #     "labels": torch.cat(target, dim=0),
        # }

        os.makedirs(save_root, exist_ok=True)
        pickle.dump(load_res, open(filename, "wb"))

    precomputed_features = load_res["patches"].to(device)
    target = load_res["labels"].to(device)
    image_features_tensor = load_res["images"].to(device)

    return precomputed_features, target, image_features_tensor


def make_descriptor_sentence(descriptor):
    if descriptor.startswith("a") or descriptor.startswith("an"):
        return f"which is {descriptor}"
    elif (
        descriptor.startswith("has")
        or descriptor.startswith("often")
        or descriptor.startswith("typically")
        or descriptor.startswith("may")
        or descriptor.startswith("can")
    ):
        return f"which {descriptor}"
    elif descriptor.startswith("used"):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"


def zeroshot_classifier(
    model,
    textnames,
    templates=None,
    is_template=True,
    make_sentence=False,
    tt_scale=None,
    device=None,
):
    with torch.no_grad():
        zeroshot_weights = []
        for i in tqdm(range(len(textnames))):
            if not is_template:
                texts = []
                for t in templates[textnames[i]]:
                    if make_sentence:
                        desc_sen = make_descriptor_sentence(t)
                        texts.append(f"{textnames[i]}, {desc_sen}")
                    else:
                        texts.append(t)
            elif templates:
                texts = [template.format(textnames[i]) for template in templates]
            else:
                texts = [f"a photo of a {textnames[i]}."]

            if i == 0:
                print(texts)

            if tt_scale is not None:
                label = f"a photo of a {textnames[i]}."
                label_tokens = clip.tokenize(label, truncate=True).to(device)
                label_embeddings = model.encode_text(label_tokens)
                label_embeddings /= label_embeddings.norm(dim=-1, keepdim=True)

            texts_tensor = clip.tokenize(texts, truncate=True).to(device)
            class_embeddings = model.encode_text(texts_tensor)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            if tt_scale is not None:  # (50,512) @ (512,1)
                weight = class_embeddings @ label_embeddings.T
                weight = (weight * tt_scale).softmax(dim=0)
                class_embedding = (class_embeddings * weight).sum(dim=0)
                class_embedding /= class_embedding.norm()
            else:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def construct_random(gpt3_prompts):
    """
    reference: https://github.com/ExplainableML/WaffleCLIP.git
    """
    key_list = list(gpt3_prompts.keys())

    # Get complete list of available descriptions.
    descr_list = [list(values) for values in gpt3_prompts.values()]
    descr_list = np.array([x for y in descr_list for x in y])

    ### Descriptor Makers.
    structured_descriptor_builder = (
        lambda item, cls: f"A photo of a {wordify(cls)}, {make_descriptor_sentence(item)}."
    )

    word_list = pickle.load(open("features/word_list.pkl", "rb"))

    avg_num_words = int(
        np.max(
            [
                np.round(np.mean([len(wordify(x).split(" ")) for x in key_list])),
                1,
            ]
        )
    )
    avg_word_length = int(
        np.round(
            np.mean(
                [np.mean([len(y) for y in wordify(x).split(" ")]) for x in key_list]
            )
        )
    )
    word_list = [x[:avg_word_length] for x in word_list]

    # (Lazy solution) Extract list of available random characters from gpt description list. Ideally we utilize a separate list.
    character_list = [x.split(" ") for x in descr_list]
    character_list = [
        x.replace(",", "").replace(".", "")
        for x in np.unique([x for y in character_list for x in y])
    ]
    character_list = np.unique(list("".join(character_list)))

    num_spaces = (
        int(np.round(np.mean([np.sum(np.array(list(x)) == " ") for x in key_list]))) + 1
    )
    num_chars = int(
        np.ceil(np.mean([np.max([len(y) for y in x.split(" ")]) for x in key_list]))
    )

    num_chars += num_spaces - num_chars % num_spaces
    sample_key = ""

    for s in range(num_spaces):
        for _ in range(num_chars // num_spaces):
            sample_key += "a"
        if s < num_spaces - 1:
            sample_key += " "

    gpt3_prompts = {key: [] for key in gpt3_prompts.keys()}

    for key in key_list:
        for _ in range(15):
            base_word = ""
            for a in range(avg_num_words):
                base_word += np.random.choice(word_list, 1, replace=False)[0]
                if a < avg_num_words - 1:
                    base_word += " "
            gpt3_prompts[key].append(structured_descriptor_builder(base_word, key))
            noise_word = ""
            use_key = sample_key if len(key) >= len(sample_key) else key
            for c in sample_key:
                if c != " ":
                    noise_word += np.random.choice(character_list, 1, replace=False)[0]
                else:
                    noise_word += ", "
            gpt3_prompts[key].append(structured_descriptor_builder(noise_word, key))

    match_key = np.random.choice(key_list)
    gpt3_prompts = {key: gpt3_prompts[match_key] for key in key_list}
    for key in gpt3_prompts:
        gpt3_prompts[key] = [
            x.replace(wordify(match_key), wordify(key)) for x in gpt3_prompts[key]
        ]

    return gpt3_prompts


def accuracy(output, target, n, dataset_name):
    # Get index of the maximum value as prediction
    if dataset_name.startswith(MyDataset.ImageNetA):
        _, pred = output[:, imagenet_a_lt].max(1)
    elif dataset_name.startswith(MyDataset.ImageNetR):
        _, pred = output[:, imagenet_r_lt].max(1)
    else:
        _, pred = output.max(1)
    # Compare prediction with target
    correct = pred.eq(target)
    # Calculate top-1 accuracy
    return float(correct.float().sum().cpu().numpy()) / n * 100

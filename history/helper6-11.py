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
from flickr30k import Flickr30kDataset, load_flickr_annotations
from edge_sam.utils.transforms import ResizeLongestSide
from torchvision.ops import masks_to_boxes

from skimage.transform import resize
from sam_sample import get_points

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


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from pytorch_wavelets import DWTForward, DWTInverse


class HighFrequencyExpert(nn.Module):
    """高频专家（傅里叶增强+局部卷积）"""

    def __init__(self, dim, filter_size=3):
        super().__init__()
        # 自适应频率阈值学习
        self.freq_threshold = nn.Parameter(torch.tensor(0.7))
        self.conv_refine = nn.Conv2d(dim, dim, kernel_size=filter_size, padding=filter_size // 2, groups=dim)

    def gaussian_highpass(self, magnitude, D0=None):
        """高斯高通滤波器"""
        if D0 is None:
            D0 = self.freq_threshold.sigmoid()  # 自适应频率阈值 [0.5-0.9]

        # 创建频率网格
        h, w = magnitude.shape[-2:]
        u = torch.linspace(-0.5, 0.5, h, device=magnitude.device)
        v = torch.linspace(-0.5, 0.5, w, device=magnitude.device)
        u, v = torch.meshgrid(u, v, indexing='ij')
        D = torch.sqrt(u ** 2 + v ** 2)

        # 高斯高通滤波器
        H = 1 - torch.exp(-(D ** 2) / (2 * D0 ** 2))
        return magnitude * H

    def forward(self, x):
        # x: [B, C, H, W]
        # 傅里叶变换
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft)

        # 分离幅度谱和相位谱
        magnitude = torch.abs(fft_shift)
        phase = torch.angle(fft_shift)

        # 高频增强
        enhanced_magnitude = self.gaussian_highpass(magnitude)

        # 重建信号
        real = enhanced_magnitude * torch.cos(phase)
        imag = enhanced_magnitude * torch.sin(phase)
        enhanced_fft = torch.complex(real, imag)
        enhanced_fft = torch.fft.ifftshift(enhanced_fft)
        enhanced_x = torch.fft.ifft2(enhanced_fft, dim=(-2, -1)).real

        # 局部卷积细化
        refined = self.conv_refine(enhanced_x)
        return (refined + x).to(orig_dtype)  # 残差连接


class WaveletMoELayer(nn.Module):
    """小波变换专家层（高低频分离处理）"""

    def __init__(self, dim, num_local_experts=4, in_features=None,
                hidden_features=None,
                act_layer=None, bias=None, drop=None,
                **kwargs):
        super().__init__()
        # 专家初始化
        self.global_expert = GlobalExpert(dim)  # 低频处理（原全局专家）
        self.hf_expert = HighFrequencyExpert(dim)  # 高频处理

        # 使用pytorch_wavelets库进行小波变换
        self.dwt_forward = DWTForward(wave='haar', mode='zero', J=1)
        self.dwt_inverse = DWTInverse(wave='haar', mode='zero')

        # 路由网络（动态资源分配）
        self.gate_global = nn.Linear(dim, 1)
        self.gate_hf = nn.Linear(dim, 1)

        # 噪声注入（增强鲁棒性）
        self.noise_std = 0.0

    def forward(self, x):
        # 分离CLS token和patch tokens
        cls_token = x[0:1, :]
        patch_tokens = x[1:, :, :]

        # 转换为空间特征 [B, C, H, W]
        N, B, C = patch_tokens.shape
        H = W = int(N ** 0.5)
        spatial_feat = patch_tokens.permute(1, 2, 0).contiguous().view(B, C, H, W)

        # Haar小波分解
        ll, coeffs = self.dwt_forward(spatial_feat)
        # 从 coeffs[0] 中提取三个方向的细节系数
        LH = coeffs[0][:, :, 0, :, :]  # 水平细节子带 [16, 768, 7, 7]
        HL = coeffs[0][:, :, 1, :, :]  # 垂直细节子带 [16, 768, 7, 7]
        HH = coeffs[0][:, :, 2, :, :]  # 对角细节子带 [16, 768, 7, 7]

        # 低频处理 (LL子带)
        ll_feat = ll.permute(0, 2, 3, 1).reshape(B, -1, C)  # 序列化 16,49,768
        ll_processed = self.global_expert(ll_feat.permute(1, 0, 2)) # N=49, B=16, C=768

        # 高频处理 (LH, HL, HH子带)
        hf_feats = []
        for hf_band in [LH, HL, HH]:
            hf_processed = self.hf_expert(hf_band)
            hf_flat = hf_processed.permute(0, 2, 3, 1).reshape(B, -1, C)
            hf_feats.append(hf_flat.permute(1, 0, 2))

        # 合并高频特征 [49, 16, 768]
        hf_combined = torch.stack(hf_feats).mean(dim=0)

        # 动态门控融合
        gate_input = torch.cat([ll_processed, hf_combined], dim=0)  # [98, 16, 768]
        gate_weights = torch.sigmoid(self.gate_global(gate_input.mean(dim=0)))

        # 添加噪声增强鲁棒性
        noise = torch.randn_like(gate_weights) * self.noise_std
        gate_weights = (gate_weights + noise).clamp(0, 1)

        # 特征融合 [49, 16, 768]
        fused_feat = (
                gate_weights * ll_processed +
                (1 - gate_weights) * hf_combined
        )

        # 重构特征图
        # 1. 转换回空间特征 [49, 16, 768] → [16, 49, 768] → [16, 768, 7, 7]
        fused_spatial = fused_feat.permute(1, 2, 0).view(B, C, 7, 7)

        # 2. 小波重构 [16, 768, 14, 14]
        highs = torch.stack([LH, HL, HH], dim=2)  # 形状变为 (N, C, 3, H, W)
        reconstructed = self.dwt_inverse((fused_spatial, [highs]))

        # 3. 转换为序列特征 [16, 768, 14, 14] → [16, 768, 196] → [196, 16, 768]
        seq_feat = reconstructed.view(B, C, -1).permute(2, 0, 1)

        # 与CLS token合并
        cls_processed = self.global_expert(cls_token)
        return torch.cat([cls_processed, seq_feat], dim=0)  # [197, 16, 768]


class LogCosineFusion(nn.Module):
    def __init__(self, dim, num_local=5):
        super().__init__()
        self.dim = dim
        self.num_local = num_local

        # 对数余弦相似度参数
        self.tau = nn.Parameter(torch.tensor(0.07))  # 可学习的温度系数
        self.log_offset = 1.0  # 对数偏移量防止数值溢出

        # 动态门控网络（保留原始结构）
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, num_local + 1)
        )
        self.temp_controller = nn.Parameter(torch.tensor(0.1))

    def log_cosine_similarity(self, global_feat, local_feats):
        """
        计算对数余弦相似度权重矩阵
        输入：
            global_feat: [B, 512]
            local_feats: [B, 5, 512]
        输出：
            weights: [B, 5] 局部特征的融合权重
        """
        # 归一化处理（关键步骤）
        global_norm = F.normalize(global_feat, p=2, dim=-1)  # [B, 512]
        local_norm = F.normalize(local_feats, p=2, dim=-1)  # [B, 5, 512]

        # 扩展全局特征 [B, 1, 512] -> [B, 5, 512]
        global_exp = global_norm.unsqueeze(1).expand_as(local_norm)

        # 计算余弦相似度 [B, 5]
        cos_sim = torch.sum(global_exp * local_norm, dim=-1)  # [1](@ref)

        # 对数变换增强区分度
        log_cos_sim = torch.log(cos_sim.clamp(min=1e-8) + self.log_offset)

        # 温度缩放与softmax归一化
        weights = F.softmax(log_cos_sim / self.tau, dim=-1)  # [B, 5]
        return weights

    def forward(self, global_feat, local_feats):
        B, K, D = local_feats.shape

        # ===== 1. 对数余弦加权融合 =====
        weights = self.log_cosine_similarity(global_feat, local_feats)  # [B, 5]
        weighted_local = torch.sum(
            local_feats * weights.unsqueeze(-1),  # [B, 5, 1] * [B, 5, 512]
            dim=1
        )  # -> [B, 512]

        # ===== 2. 动态门控加权 =====
        global_expanded = global_feat.unsqueeze(1).expand(-1, K, -1)
        gate_input = torch.cat([global_expanded, local_feats], dim=-1)
        gate_logits = self.gate_net(gate_input)
        gate_weights = F.softmax(gate_logits / self.temp_controller, dim=-1)

        # 分割权重
        global_weights = gate_weights[..., 0]  # [B, 5]
        local_weights = gate_weights[..., 1]  # [B, 5]

        # 特征加权
        global_component = torch.sum(
            global_weights.unsqueeze(-1) * global_feat.unsqueeze(1),
            dim=1
        )
        local_component = torch.sum(
            local_weights.unsqueeze(-1) * local_feats,
            dim=1
        )
        gated_fusion = global_component + local_component  # [B, 512]

        # ===== 3. 残差融合 =====
        return weighted_local + gated_fusion  # [B, 512]


class GOAL_CLIP(nn.Module):
    def __init__(self, clip_model, device, num_local_experts=4, num_crops=5):
        super().__init__()
        # 初始化CLIP骨干
        self.visual = clip_model.visual
        self.text_encoder = clip_model.transformer
        self.token_embed = clip_model.token_embedding
        self.positional_embed = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.num_crops = num_crops
        self.device = device
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 冻结CLIP所有参数
        # for param in self.visual.parameters():
        #     param.requires_grad_(False)
        # 0.0540 | 图像->文本  Top - 5: 0.2592
        for name, param in self.visual.named_parameters():
            if "blocks.11" in name or "blocks.10" in name or "blocks.9" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        for param in self.text_encoder.parameters():
            param.requires_grad_(False)
        for param in [self.token_embed, self.positional_embed,
                      self.ln_final, self.text_projection]:
            param.requires_grad_(False)

        # 替换ViT中的MLP层为小波MoE层
        # for i in range(len(self.visual.transformer.resblocks)):
        #     # 仅在中间层应用小波变换
        #     if 2 <= i <= 8:  # 选择中间层
        #         original_block = self.visual.transformer.resblocks[i]
        #         dim = original_block.mlp[0].in_features
        #         new_block = Block(
        #             dim, num_heads=original_block.attn.num_heads,
        #             mlp_layer=partial(WaveletMoELayer, dim=dim, num_local_experts=num_local_experts),
        #             qkv_bias=True
        #         ).to(self.dtype)
        #         self.visual.transformer.resblocks[i] = new_block

        # 新增融合控制器（全局+5个局部）
        self.lc_fusion = LogCosineFusion(dim=512, num_local=5)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, images):
        """
        提取图像特征（支持多裁剪图处理）
        输入: [B, num_crops+1, C, H, W]
        输出: [B, 256] 融合后的图像特征
        """
        batch = images.shape[0]
        global_images = images[:, 0, ...]
        crop_images = images[:, 1:, ...].reshape(-1, *images.shape[-3:])

        # 全局特征提取
        global_feat = self.visual(global_images.type(self.dtype))

        # 局部特征提取
        crop_feats = self.visual(crop_images.type(self.dtype))
        crop_feats = crop_feats.reshape(batch, self.num_crops, -1)

        # 特征融合
        fusion_features = self.fuse_features(global_feat, crop_feats)

        return fusion_features

    def encode_text(self, texts):
        """
        提取文本特征（支持多描述处理）
        输入: [B, num_descriptions, seq_len]
        输出: [B, num_descriptions, 512] 文本特征
        """
        text_feats = []
        for i in range(texts.shape[1]):
            text_feat = self.encode_single_text(texts[:, i])
            text_feats.append(text_feat)
        return torch.stack(text_feats, dim=1)

    def encode_single_text(self, text):
        """处理单文本描述"""
        x = self.token_embed(text).type(self.dtype)
        x = x + self.positional_embed.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.text_encoder(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        # 取EOS位置特征
        eos_features = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        projected = eos_features @ self.text_projection
        return projected

    def forward(self, images, texts):
        # 图像特征提取
        image_features = self.encode_image(images)  # [B, 512]
        # 文本特征提取
        text_features = self.encode_text(texts)  # [B, 5, 512]
        # 归一化特征
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        # 展平文本特征 [B, 5, 256] -> [B*5, 256]
        text_features_flat = text_features.reshape(-1, text_features.shape[-1])

        # 计算相似度矩阵
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features_flat.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def fuse_features(self, global_feat, local_feats):
        """
        融合全局特征与多局部特征（支持动态门控与残差连接）
        输入：
            global_feat : [B, 512]       # 全局CLIP特征
            local_feats : [B,5,512]      # SAM提取的5个局部区域CLIP特征
        输出：
            combined_feat : [B, 512]     # 融合后的对比学习特征
        """
        return self.lc_fusion(global_feat, local_feats)
        # return global_feat

    # 对比损失函数
    def multi_positive_contrastive_loss(self, logits_per_image, logits_per_text, temperature=0.07):
        """
        多正样本对比损失函数（支持Flickr多描述特性）
        输入:
            logits_per_image: [batch_size, batch_size*5] 图像到所有文本相似度
            logits_per_text: [batch_size*5, batch_size] 文本到所有图像相似度
        输出:
            标量损失值
        """
        device = logits_per_image.device
        B = logits_per_image.size(0)  # 批次大小
        num_captions = 5  # 每个图像的文本描述数

        # ===== 图像到文本损失 =====
        # 创建目标分布: [B, B*5] 对角线区块为1/5
        img_target = torch.zeros(B, B * num_captions, device=device)
        for i in range(B):
            img_target[i, i * num_captions:(i + 1) * num_captions] = 1.0 / num_captions

        # KL散度需要log_softmax输入
        img_logits = logits_per_image / temperature
        img_logs = F.log_softmax(img_logits, dim=-1)
        img_loss = F.kl_div(img_logs, img_target, reduction='batchmean')

        # ===== 文本到图像损失 =====
        # 创建标签: [B*5] 每个文本描述对应的图像索引
        txt_labels = torch.arange(B, device=device).repeat_interleave(num_captions)

        # 交叉熵损失
        txt_logits = logits_per_text / temperature
        txt_loss = F.cross_entropy(txt_logits, txt_labels)

        return (img_loss + txt_loss) / 2


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
        # 语义置信度计算
        uncertainty = torch.sigmoid(self.uncertainty_proj(text_emb))  # (B,L,1)

        # 跨模态语义路由
        refined_emb, _ = self.semantic_router(
            text_emb, image_emb, image_emb,
            key_padding_mask=(uncertainty < 0.5).squeeze()
        )
        return refined_emb * uncertainty

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
    save_root = f"weights/{dataset_name}"

    # if save_root not exist, create it
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    filename = os.path.join(save_root, f"{save_file}-{alpha}-{n_samples}.pkl")

    # 定义主函数需要的三个返回值
    precomputed_features = None
    image_paths = []
    all_captions = []

    if os.path.exists(filename):
        print(f"Loading {filename}...")
        # load_res = pickle.load(open(filename, "rb"))
        # precomputed_features, image_paths, all_captions = load_res
        # return precomputed_features, image_paths, all_captions
        model = GOAL_CLIP(clip_model=model, device=device)
        model.load_state_dict(torch.load(filename))
        return model, image_paths, all_captions
    else:
        print(f"File {filename} not found, precomputing features...")

        annotation_path = '/home/wuziyi/Project/Flickr/Flickr30K_Entities/data/results_20130124.token'  # 替换为实际路径
        annotations_df = load_flickr_annotations(annotation_path)
        image_dir = '/home/wuziyi/Project/Flickr/Flickr30K_Entities/data/flickr30k-images'  # 替换为实际路径
        dataset = Flickr30kDataset(
            image_dir=image_dir,
            annotation_df=annotations_df,
            num_crops=5
        )

        # 获取图像路径和描述
        image_paths = dataset.image_paths
        all_captions = dataset.captions

        train_loader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            collate_fn=lambda batch: [torch.stack([item[0] for item in batch]),
                                      torch.stack([item[1] for item in batch])],
            num_workers=num_workers,
            pin_memory=True
        )

        num_epochs = 5
        # 创建GOAL_CLIP模型
        model = GOAL_CLIP(
            clip_model=model,
            device=device,
            num_local_experts=4,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-5,
            weight_decay=0.01
        )


        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            epoch_losses = []
            for batch_idx, (images, texts) in tqdm(enumerate(train_loader)):
                with torch.cuda.amp.autocast(enabled=True):
                    # images = B=16, NS=10, C=3, H=224, W=224
                    texts = texts.to(device)
                    images = images.to(device)

                    b, ns = images.shape[:2]

                    # 前向传播
                    logits_img, logits_txt = model(images, texts)

                    # 计算损失
                    with torch.cuda.amp.autocast(enabled=False):  # 关键计算切回 FP32
                        logits_img = logits_img.float()
                        logits_txt = logits_txt.float()
                        loss = model.multi_positive_contrastive_loss(logits_img, logits_txt)

                    # 记录损失
                    epoch_losses.append(loss.item())
                    total_loss += loss.item()

                    if torch.isnan(loss):
                        print(f"Bad sample at index {batch_idx}")  # 记录问题样本索引
                        for name, param in model.named_parameters():
                            if param.grad is not None and torch.isnan(param.grad).any():
                                print(f"NaN gradient in {name}")
                                exit(0)

                    optimizer.zero_grad()
                    loss.backward()
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    # 参数更新
                    optimizer.step()

                    if batch_idx % 100 == 0:
                        print(f"Epoch {epoch + 1}/{num_epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")

            # 记录每轮平均损失
            avg_epoch_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

        # 保存模型
        # torch.save(model.state_dict(), model_save_path)
        torch.save(model.state_dict(), filename)
        print(f"Model saved to {filename}")
        return model, image_paths, all_captions  # 直接返回训练好的模型

        # # 计算并保存特征
        # print("Computing features for the entire dataset...")
        # all_features = []
        # model.eval()
        # with torch.no_grad():
        #     for images, texts in tqdm(train_loader):
        #         with torch.cuda.amp.autocast(enabled=True):
        #             images = images.to(device)
        #             # 提取图像特征
        #             global_feat = model.visual(images[:, 0, ...].type(model.dtype))
        #             crop_feats = model.visual(images[:, 1:, ...].reshape(-1, *images.shape[-3:]).type(model.dtype))
        #             crop_feats = crop_feats.reshape(images.shape[0], model.num_crops, -1)
        #             combined_feat = model.fuse_features(global_feat, crop_feats)
        #             all_features.append(combined_feat.cpu())
        #
        # precomputed_features = torch.cat(all_features, dim=0)
        # # 保存特征、路径和描述
        # save_data = (precomputed_features, image_paths, all_captions)
        # with open(filename, 'wb') as f:
        #     pickle.dump(save_data, f)
        # print(f"Features saved to {filename}")
        #
        # # 添加评估指标（可选）
        # # results['evaluation_metrics'] = evaluate_model(model, dataset)
        # return precomputed_features, image_paths, all_captions


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

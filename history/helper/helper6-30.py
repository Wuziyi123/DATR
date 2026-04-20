import gc
import os
import random

import torch.fft
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
import pywt
from transformers import BertModel, BertConfig, BertTokenizer
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
import torch.nn as nn
from timm.models.vision_transformer import Block
from functools import partial
from flickr30k import Flickr30kDataset, load_flickr_annotations
from torchvision.ops import masks_to_boxes
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


class SemanticEmbeddingLearner(nn.Module):
    """语义嵌入学习器：通过跨模态注意力增强视觉-语义对应"""
    def __init__(self, embed_dim, num_semantic_tokens=8, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_semantic_tokens = num_semantic_tokens

        # 可学习的语义标记
        self.semantic_tokens = nn.Parameter(torch.randn(1, num_semantic_tokens, embed_dim))
        nn.init.xavier_uniform_(self.semantic_tokens)

        # 跨模态注意力层
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        # 语义引导的标记注意力
        self.token_attention = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )

        # 门控融合机制
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )

        # 正交约束损失
        self.ortho_loss_weight = 0.1

    def forward(self, visual_feats, text_feats):
        """
        visual_feats: [B, N, D] 视觉特征
        text_feats: [B, M, D] 文本特征
        返回: 增强的视觉特征和正则化损失
        """
        B, N, D = visual_feats.shape
        M = text_feats.size(1)

        # 扩展语义标记
        semantic_tokens = self.semantic_tokens.expand(B, -1, -1)

        # 跨模态注意力：文本引导的语义增强
        attn_output, _ = self.cross_attn(
            query=semantic_tokens,
            key=text_feats,
            value=text_feats
        )
        visual_semantic = attn_output  # [B, num_semantic_tokens, D]

        # 语义引导的视觉标记注意力
        attn_scores = []
        for i in range(N):
            # 每个视觉标记与所有语义标记的交互
            visual_token = visual_feats[:, i].unsqueeze(1)  # [B, 1, D]
            expanded_visual = visual_token.expand(-1, self.num_semantic_tokens, -1)  # [B, num_semantic_tokens, D]

            # 计算注意力分数
            similarity_input = torch.cat([
                expanded_visual,
                visual_semantic,
                expanded_visual * visual_semantic
            ], dim=-1)
            score = self.token_attention(similarity_input)  # [B, num_semantic_tokens, 1]
            attn_scores.append(score.squeeze(-1))

        attn_scores = torch.stack(attn_scores, dim=1)  # [B, N, num_semantic_tokens]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N, num_semantic_tokens]

        # 语义聚合
        semantic_aggregated = torch.einsum('bnk,bkd->bnd', attn_weights, visual_semantic)

        # 门控融合
        gate_input = torch.cat([
            visual_feats,
            semantic_aggregated,
            visual_feats - semantic_aggregated
        ], dim=-1)
        gate_values = self.gate_network(gate_input)
        enhanced_visual = gate_values * visual_feats + (1 - gate_values) * semantic_aggregated

        # 正交约束
        enhanced_norm = F.normalize(enhanced_visual, p=2, dim=-1)  # L2归一化
        ortho_loss = torch.norm(
            torch.matmul(enhanced_norm, enhanced_norm.transpose(1, 2)) -
            torch.eye(N).to(enhanced_norm.device),
            p='fro'
        )

        # 特征协方差冗余损失
        feat_matrix = enhanced_visual.reshape(B * N, D)  # [B*N, D]
        feat_matrix = feat_matrix - feat_matrix.mean(dim=0)  # 中心化
        feat_matrix = F.normalize(feat_matrix, p=2, dim=-1)  # L2归一化
        # 计算协方差矩阵
        cov_matrix = feat_matrix.T @ feat_matrix / (B * N - 1)  # [D, D]
        # COR损失：最小化非对角线元素
        off_diag_mask = ~torch.eye(D, dtype=bool)  # 非对角线掩码
        redundancy_loss = torch.norm(cov_matrix[off_diag_mask], p=2)  # 非对角线L2范数
        # 数值保护（防止极端值）
        redundancy_loss = torch.clamp(redundancy_loss, max=10.0)  # 截断上限

        total_reg_loss = redundancy_loss + self.ortho_loss_weight * ortho_loss

        return enhanced_visual, total_reg_loss


class FourierEnhancement(nn.Module):
    """傅里叶增强：聚焦低频结构信息"""

    def forward(self, x):
        # 傅里叶变换
        f = torch.fft.fft2(x)
        fshift = torch.fft.fftshift(f)

        # 创建低通滤波器
        _, _, h, w = x.shape
        mask = torch.zeros(h, w).to(x.device)
        cx, cy = h // 2, w // 2
        mask[cx - 30:cx + 30, cy - 30:cy + 30] = 1.0

        # 应用低通滤波
        magnitude = torch.abs(fshift) * mask
        phase = torch.angle(fshift)

        # 重构图像
        f_ishift = torch.fft.ifftshift(magnitude * torch.exp(1j * phase))
        img_recon = torch.fft.ifft2(f_ishift)
        return torch.abs(img_recon)


class WaveletEnhancement(nn.Module):
    """小波增强：聚焦高频细节信息（含软阈值降噪）"""
    def __init__(self, wavelet='bior4.4', threshold=0.15):
        super().__init__()
        self.wavelet = wavelet
        self.threshold = threshold  # 软阈值参数
        self.enhance_scales = [2.0, 1.8, 1.5]  # 对应HH, HL, LH方向的增强系数

    def forward(self, x):
        """
        输入: x (torch.Tensor) 形状 [B, 3, H, W]
        输出: 增强后的图像 (torch.Tensor) 形状 [B, 3, H, W]
        """
        batch_size, channels, height, width = x.shape
        processed = []

        # 将张量转换为CPU上的NumPy数组（假设CLIP在CPU上）
        x_np = x.cpu().permute(0, 2, 3, 1).numpy()  # [B, H, W, 3]

        for c in range(channels):
            channel_data = x_np[:, :, :, c]  # [B, H, W]
            enhanced_channel = []

            for b in range(batch_size):
                # 单样本单通道处理
                sample = channel_data[b]

                # 小波分解（二维单层）
                coeffs = pywt.dwt2(sample, self.wavelet)
                cA, (cH, cV, cD) = coeffs

                # 软阈值降噪（同时处理三个高频分量）
                cH = pywt.threshold(cH, self.threshold * np.std(cH), mode='soft')
                cV = pywt.threshold(cV, self.threshold * np.std(cV), mode='soft')
                cD = pywt.threshold(cD, self.threshold * np.std(cD), mode='soft')

                # 方向敏感增强
                cH = np.clip(cH * self.enhance_scales[0], -1, 1)
                cV = np.clip(cV * self.enhance_scales[1], -1, 1)
                cD = np.clip(cD * self.enhance_scales[2], -1, 1)

                # 小波重构
                reconstructed = pywt.idwt2((cA, (cH, cV, cD)), self.wavelet)

                # 尺寸对齐（处理偶数尺寸问题）
                if reconstructed.shape != sample.shape:
                    pad_h = (0, sample.shape[0] - reconstructed.shape[0])
                    pad_w = (0, sample.shape[1] - reconstructed.shape[1])
                    reconstructed = np.pad(reconstructed, (pad_h, pad_w), mode='edge')

                enhanced_channel.append(reconstructed)

            processed.append(np.stack(enhanced_channel, axis=0))  # [B, H, W]

        # 合并通道并转换回张量
        processed = np.stack(processed, axis=-1)  # [B, H, W, 3]
        wave_crop = torch.tensor(processed).permute(0, 3, 1, 2).to(x.device)  # [B, 3, H, W]

        return wave_crop


# ===== 核心模块实现 =====

class GlobalFourierPath(nn.Module):
    """全局傅里叶路径：增强结构信息"""

    def __init__(self, clip_model, embed_dim):
        super().__init__()
        self.clip = clip_model
        self.embed_dim = embed_dim

        # 傅里叶变换
        self.fourier = FourierEnhancement()

        # 低频增强模块
        self.low_freq_enhancer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # CLIP特征增强
        self.clip_enhancer = nn.Sequential(
            nn.Linear(512, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, global_img):
        # 原始CLIP特征
        with torch.no_grad():
            orig_clip_feat = self.clip.encode_image(global_img)

        # 傅里叶增强
        fourier_img = self.fourier(global_img)

        # 低频特征提取
        low_freq_feat = self.low_freq_enhancer(fourier_img).flatten(1)

        # 傅里叶增强的CLIP特征
        with torch.no_grad():
            fourier_clip_feat = self.clip.encode_image(fourier_img)

        # 特征融合
        fused_feat = torch.cat([
            self.clip_enhancer(orig_clip_feat.float()),
            self.clip_enhancer(fourier_clip_feat.float()),
            low_freq_feat
        ], dim=1)

        return nn.Linear(fused_feat.size(1), self.embed_dim, device=fused_feat.device)(fused_feat)


class LocalWaveletPath(nn.Module):
    """局部小波路径：增强细节信息"""
    def __init__(self, clip_model, embed_dim, num_crops):
        super().__init__()
        self.clip = clip_model
        self.embed_dim = embed_dim
        self.num_crops = num_crops

        # 小波增强
        self.wavelet = WaveletEnhancement(threshold=0.15)

        # 高频增强模块
        self.high_freq_enhancer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((7, 7))
        )

    def forward(self, crops):
        """crops: [B, num_crops, 3, 224, 224]"""
        batch_size = crops.size(0)
        enhanced_feats = []

        for i in range(self.num_crops):
            crop = crops[:, i]

            # 小波增强
            wave_crop = self.wavelet(crop)

            # 原始CLIP特征
            with torch.no_grad():
                orig_feat = self.clip.encode_image(crop)

            # 小波增强的CLIP特征
            with torch.no_grad():
                wave_feat = self.clip.encode_image(wave_crop)

            # 高频特征提取
            high_freq_feat = self.high_freq_enhancer(wave_crop).flatten(1)

            # 特征融合
            fused_feat = torch.cat([
                orig_feat.float(),
                wave_feat.float(),
                high_freq_feat
            ], dim=1)

            enhanced_feats.append(nn.Linear(fused_feat.size(1), self.embed_dim, device=fused_feat.device)(fused_feat))

        return torch.stack(enhanced_feats, dim=1)


class ContextEnhancer(nn.Module):
    """场景上下文增强模块"""
    def __init__(self, embed_dim):
        super().__init__()
        # 场景理解Transformer
        self.context_transformer = nn.Sequential(
            Block(embed_dim*2, num_heads=8),
            Block(embed_dim*2, num_heads=8)
        )

        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(embed_dim*2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, global_feat, local_feats):
        # 将全局特征转换为空间形式
        B, D = global_feat.shape
        global_spatial = global_feat.view(B, D, 1, 1).expand(-1, -1, 7, 7)

        # 将局部特征聚合为空间图
        local_aggregated = torch.mean(local_feats, dim=1).view(B, D, 1, 1).expand(-1, -1, 7, 7)

        # 场景特征融合
        scene_feats = torch.cat([global_spatial, local_aggregated], dim=1)

        # 上下文理解
        scene_feats = scene_feats.flatten(2).permute(0, 2, 1)  # [B, 49, 2D]
        scene_feats = self.context_transformer(scene_feats)
        scene_feats = scene_feats.permute(0, 2, 1).view(B, -1, 7, 7)

        # 空间注意力加权
        attn_weights = self.spatial_attn(scene_feats)
        enhanced_feats = attn_weights * scene_feats

        # 全局池化
        return torch.mean(enhanced_feats, dim=[2, 3])


class TextEncoder(nn.Module):
    """文本编码器：BERT+语义增强"""

    def __init__(self, embed_dim):
        super().__init__()
        self.bert = BertModel(BertConfig(
            hidden_size=embed_dim,
            num_hidden_layers=4,
            num_attention_heads=8
        ))

        # 语义增强
        self.semantic_enhancer = nn.Sequential(
            Block(embed_dim, num_heads=8),
            Block(embed_dim, num_heads=8)
        )

    def forward(self, texts, attention_mask):
        outputs = self.bert(input_ids=texts, attention_mask=attention_mask)
        text_feats = outputs.last_hidden_state
        return self.semantic_enhancer(text_feats)


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
        cos_sim = torch.sum(global_exp * local_norm, dim=-1)

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
    def __init__(self, clip_model, device, num_local_experts=4, num_crops=20,
                 embed_dim=512, num_semantic_tokens=8):
        super().__init__()
        # 初始化CLIP骨干
        self.embed_dim = embed_dim
        self.visual = clip_model.visual
        self.text_encoder = clip_model.transformer
        self.token_embed = clip_model.token_embedding
        self.positional_embed = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.num_crops = num_crops
        self.device = device
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # 投影层
        self.image_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 256)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 256)
        )
        # === 全局特征路径：傅里叶增强 ===
        self.global_path = GlobalFourierPath(clip_model, embed_dim)

        # === 局部特征路径：小波增强 ===
        self.local_path = LocalWaveletPath(clip_model, embed_dim, num_crops)

        # === 上下文场景增强模块 ===
        self.context_enhancer = ContextEnhancer(embed_dim)

        # === 语义嵌入学习器 ===
        self.semantic_learner = SemanticEmbeddingLearner(
            embed_dim,
            num_semantic_tokens=num_semantic_tokens
        )
        # self.image_proj = nn.Linear(512, 256)
        # self.text_proj = nn.Linear(512, 256)

        # 冻结CLIP所有参数
        # for param in self.visual.parameters():
        #     param.requires_grad_(False)
        # 图像->文本 Top-1: 0.31 | 图像->文本 Top-5: 0.75
        # Top-1: 0.26 | 图像->文本 Top-5: 0.73

        # for name, param in self.visual.named_parameters():
        #     if "blocks.11" in name or "blocks.10" in name or "blocks.9" in name:
        #         param.requires_grad_(True)
        #     else:
        #         param.requires_grad_(False)

        # 文本部分保持冻结
        for param in self.text_encoder.parameters():
            param.requires_grad_(False)
        for param in [self.token_embed, self.positional_embed, self.ln_final]:
            param.requires_grad_(False)

        # 文本增强
        self.text_augmentation = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 512)
        )

        # 替换ViT中的MLP层为小波MoE层
        # for i in range(len(self.visual.transformer.resblocks)):
        #     # 仅在中间层应用小波变换
        #     if 4 <= i <= 8:  # 选择中间层
        #         original_block = self.visual.transformer.resblocks[i]
        #         dim = original_block.mlp[0].in_features
        #         new_block = Block(
        #             dim, num_heads=original_block.attn.num_heads,
        #             mlp_layer=partial(WaveletMoELayer, dim=dim, num_local_experts=num_local_experts),
        #             qkv_bias=True
        #         ).to(self.dtype)
        #         self.visual.transformer.resblocks[i] = new_block

        # 新增融合控制器（全局+5个局部）
        self.lc_fusion = LogCosineFusion(512, self.num_crops)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, images, text_features):
        """
        提取图像特征（支持多裁剪图处理）
        输入: [B, num_crops+1, C, H, W]
        输出: [B, 512] 融合后的图像特征
        """
        # 全局特征 (傅里叶增强)
        global_feat = self.global_path(images[:, 0])
        # 局部特征 (小波增强)
        local_feats = self.local_path(images[:, 1:1 + self.num_crops])
        # ===== 语义嵌入学习 =====
        # 视觉特征语义增强
        enhanced_local, reg_loss = self.semantic_learner(local_feats, text_features)
        # 场景上下文增强
        context_feat = self.context_enhancer(global_feat, enhanced_local)

        return context_feat, reg_loss

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
        return torch.stack(text_feats, dim=1).mean(dim=1)

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
        # 文本特征提取
        text_features = self.encode_text(texts)  # [B, 5, 512]
        # 图像特征提取
        image_features, reg_loss = self.encode_image(images, texts)  # [B, 512]

        # 投影到共享空间
        image_features = self.image_proj(image_features)
        text_features = self.text_proj(text_features)

        loss = self.multi_positive_contrastive_loss(image_features, text_features)
        loss = loss + reg_loss

        return loss

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
    def multi_positive_contrastive_loss(self, img_feats, text_feats, ):
        # 归一化特征
        img_feats = F.normalize(img_feats, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)
        # 计算相似度
        logit_scale = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)
        logits_per_image = logit_scale * img_feats @ text_feats.t()
        logits_per_text = logit_scale * text_feats @ img_feats.t()

        # 对比目标
        labels = torch.arange(img_feats.size(0), device=img_feats.device)

        # 对比损失
        img_loss = F.cross_entropy(logits_per_image, labels)
        txt_loss = F.cross_entropy(logits_per_text, labels)

        # 总损失
        total_loss = (img_loss + txt_loss)/2

        return total_loss


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
            num_crops=20
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

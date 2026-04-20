import torch
import torch.nn as nn
import numpy as np
import pywt
from torch.fft import fftn, ifftn
import math
import json
import os
import pickle
import fire
import yaml
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from flickr30k import load_flickr_annotations, Flickr30kDataset
from helper import set_seed
import sys
from typing import Tuple, List
from torch.nn import functional as F
from PIL import Image
import clip
from torch.optim import AdamW

# 设置随机种子
set_seed(0)

# 全局标注字典
annotation_dict = {}


# --------------------- 小波-RKHS融合模块  ---------------------
class WaveletRKHSOperator(nn.Module):
    """小波-希尔伯特细节增强算子 (泛函分析应用)"""

    def __init__(self, in_channels, wavelet='db4', levels=2, sigma=1.0):
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.sigma = sigma

        # 随机傅里叶特征 (RFF) 映射
        self.rff_dim = 128
        self.rff_weights = nn.Parameter(torch.randn(in_channels, self.rff_dim // 2) * (2 * sigma ** 2))

        # 多尺度小波卷积 - 输入通道数应为3 (cH, cV, cD)
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(3, in_channels, 3, padding=1)  # 改为3输入通道
            for _ in range(levels)
        ])

    def _rff_mapping(self, x):
        """随机傅里叶特征映射"""
        # 高斯核逼近: k(x,y)=exp(-||x-y||^2/(2σ^2))
        B, C, H, W = x.shape
        proj = torch.matmul(x.view(B, C, -1).permute(0, 2, 1), self.rff_weights)
        # 傅里叶特征构造
        cos_feat = torch.cos(proj)
        sin_feat = torch.sin(proj)
        return torch.cat([cos_feat, sin_feat], dim=-1).view(B, -1, H, W)

    def _hilbert_transform(self, x):
        """希尔伯特变换 (增强边缘特征)"""
        orig_dtype = x.dtype  # 保存原始数据类型
        # 转换为float32以支持任意尺寸的FFT
        x = x.to(torch.float32)
        f = fftn(x, dim=(-2, -1))
        h = torch.zeros_like(f)
        h[..., :x.size(-2) // 2, :x.size(-1) // 2] = 1
        h[..., -x.size(-2) // 2:, -x.size(-1) // 2:] = 1
        result = ifftn(f * h, dim=(-2, -1)).real
        # 转换回原始数据类型
        return result.to(orig_dtype)

    def forward(self, x):
        B, C, H, W = x.shape
        wavelet_feats = []

        # 希尔伯特变换增强纹理
        hilbert_feat = self._hilbert_transform(x)

        # 多级小波分解
        for level in range(2, self.levels + 1):
            level_feats = []
            # 对每个样本单独处理小波分解
            for i in range(hilbert_feat.shape[0]):
                # 保存当前数据类型
                orig_dtype = hilbert_feat.dtype
                # 分离出当前样本并移除通道维度
                hilbert_sample = hilbert_feat[i].squeeze(0)  # [H, W]
                # 分离计算图并转换为NumPy
                hilbert_cpu = hilbert_sample.detach().to(torch.float32).cpu().numpy()

                # 小波分解
                coeffs = pywt.wavedec2(hilbert_cpu, self.wavelet, level = level)
                # 取第一组高频系数
                cH, cV, cD = coeffs[1]

                # 创建特征图并调整维度
                detail_map = np.stack([cH, cV, cD], axis=0)  # [3, h, w]
                level_feats.append(detail_map)

            # 组合所有样本的特征图
            detail_feat = np.stack(level_feats, axis=0)  # [B, 3, h, w]
            detail_tensor = torch.tensor(detail_feat, device=x.device).to(orig_dtype)

            # 小波卷积增强
            detail_conv = self.wavelet_convs[level - 1](detail_tensor)
            # 上采样回原始尺寸
            wavelet_feats.append(F.interpolate(detail_conv, (H, W)))

        # RKHS映射
        concatenated = torch.cat(wavelet_feats, dim=1)  # 连接不同层级的特征
        averaged = concatenated.mean(dim=1, keepdim=True)  # 平均所有层级
        rff_feat = self._rff_mapping(averaged)  # 应用RFF映射

        return rff_feat


# --------------------- 金字塔注意力融合 ---------------------
class PyramidalAttentionFusion(nn.Module):
    """金字塔注意力融合 (多尺度特征交互)"""

    def __init__(self, dim, scales=[1.0, 0.5, 0.25], num_heads=8):
        super().__init__()
        self.scales = scales
        self.num_heads = num_heads

        # 多尺度卷积
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=int(1 / s))
            for s in scales
        ])

        # 跨尺度注意力
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        # Sobolev约束层
        self.sobolev_norm = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh()
        )

    def forward(self, features):
        """多尺度特征融合"""
        # 多尺度特征提取
        scale_feats = []
        for i, scale in enumerate(self.scales):
            resized = F.interpolate(features, scale_factor=scale, mode='bilinear')
            conv_feat = self.conv_layers[i](resized)
            # 恢复原始尺寸
            scale_feats.append(F.interpolate(conv_feat, features.shape[-2:]))

        # 跨尺度注意力机制
        B, C, H, W = features.shape
        q = self.query(features.flatten(2).permute(0, 2, 1))
        attended_feats = []
        for feat in scale_feats:
            k = self.key(feat.flatten(2).permute(0, 2, 1))
            v = self.value(feat.flatten(2).permute(0, 2, 1))

            # 注意力计算
            attn_logits = torch.matmul(q, k.permute(0, 2, 1)) / (self.num_heads ** 0.5)
            attn_weights = F.softmax(attn_logits, dim=-1)
            attended = torch.matmul(attn_weights, v)
            attended_feats.append(attended)

        # 特征融合与Sobolev约束
        combined = sum(attended_feats) / len(attended_feats)
        # 应用Sobolev约束
        normalized = self.sobolev_norm(combined)
        return normalized.view(B, C, H, W)


# --------------------- 专家网络与融合模块 ---------------------
class OrthogonalExpertLayer(nn.Module):
    """正交约束专家层 (移除了因果调制)"""

    def __init__(self, dim, num_experts=4, ortho_weight=0.1):
        super().__init__()
        self.ortho_weight = ortho_weight
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(num_experts)
        ])

        # 路由网络
        self.router = nn.Sequential(
            nn.Linear(dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def _orthogonal_loss(self, expert_outputs):
        """专家正交约束损失"""
        # 计算专家特征的相关性
        features = torch.stack(expert_outputs, dim=1)  # [B, K, D]
        cov = torch.matmul(features.permute(0, 2, 1), features)  # [B, D, D]
        identity = torch.eye(features.size(2), device=features.device).unsqueeze(0)
        return F.mse_loss(cov, identity.expand_as(cov))

    def forward(self, x):
        # 动态路由
        routing_weights = self.router(x)  # x应为[B, dim]

        # 专家计算
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)
            expert_outputs.append(expert_out)

        # 正交约束损失
        ortho_loss = self._orthogonal_loss(expert_outputs)

        # 加权融合
        weighted_outs = []
        for i, out in enumerate(expert_outputs):
            weight = routing_weights[:, i].unsqueeze(-1)  # [B, 1]
            weighted_outs.append(out * weight)
        combined_expert = sum(weighted_outs)  # [B, dim]

        return combined_expert, self.ortho_weight * ortho_loss


# --------------------- 完整网络架构 (移除了因果模块) ---------------------
class MathEnhancedGOALCLIP(nn.Module):
    """数学增强的多模态对比网络"""

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

        # 冻结CLIP参数
        # 解冻CLIP后3层
        for name, param in self.visual.named_parameters():
            if "blocks.11" in name or "blocks.10" in name or "blocks.9" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        # 文本部分保持冻结
        for param in self.text_encoder.parameters():
            param.requires_grad_(False)
        for param in [self.token_embed, self.positional_embed, self.ln_final, self.text_projection]:
            param.requires_grad_(False)

        # 增强的视觉处理路径
        # 使用线性层重塑特征为可视为空间特征的维度
        self.feature_reshape = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 512 * 2),
            nn.GELU(),
            nn.Linear(512 * 2, 576)
        )

        self.wavelet_rkhs = WaveletRKHSOperator(1)  # 输入通道数为1

        # 金字塔注意力融合
        self.pyramidal_attn = PyramidalAttentionFusion(128)

        # 添加特征适配层，将128维特征扩展到512维
        self.feature_adaptor = nn.Sequential(
            nn.Linear(128, 512),
            nn.GELU()
        )

        # 移除因果推断模块

        # 专家网络 (移除了因果调制)
        self.expert_layer = OrthogonalExpertLayer(512, num_experts=num_local_experts)

        # 文本增强
        self.text_augmentation = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 512)
        )

        # 投影层
        self.image_proj = nn.Linear(512, 256)
        self.text_proj = nn.Linear(512, 256)

        # 损失函数参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.temp = nn.Parameter(torch.ones([]) * 0.07)

    def encode_image(self, images):
        """多模态图像处理路径"""
        batch = images.shape[0]
        global_img = images[:, 0, ...]
        crop_imgs = images[:, 1:, ...].reshape(-1, *images.shape[-3:])

        # 全局特征处理
        global_feat = self.visual(global_img.type(self.dtype))
        global_feat = global_feat.unsqueeze(1)  # [B, 1, D]

        # 局部特征处理
        crop_feats = self.visual(crop_imgs.type(self.dtype))  # [B*num_crops, D]
        crop_feats = crop_feats.reshape(batch, self.num_crops, -1)  # [B, num_crops, D]

        # 组合全局和局部特征
        all_feats = torch.cat([global_feat, crop_feats], dim=1)  # [B, N, D]
        B, N, D = all_feats.shape

        # 重塑特征为可视为空间特征的格式
        spatial_feats = self.feature_reshape(all_feats)  # [B, N, 576]
        spatial_feats = spatial_feats.view(B, N, 24, 24)  # [B, N, H, W]

        # 重新排列维度以适应小波变换 [B, N, H, W] -> [B*N, 1, H, W]
        spatial_feats = spatial_feats.view(B * N, 1, 24, 24)

        # 应用小波-RKHS操作
        enhanced_feats = self.wavelet_rkhs(spatial_feats)  # [B*N, C, H, W]

        # 恢复原始结构 [B*N, C, H, W] -> [B, N, C, H, W]
        _, C, H, W = enhanced_feats.shape
        enhanced_feats = enhanced_feats.view(B, N, C, H, W)  # [B, N, C, H, W]

        # 计算每张图片的总像素数
        total_pixels_per_image = N * H * W

        # 计算最近的平方尺寸
        closest_side = int(math.isqrt(total_pixels_per_image))
        closest_square = closest_side * closest_side

        # 确保形状变换不会改变元素数量
        reshaped_feats = enhanced_feats.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, N, H, W]
        reshaped_feats = reshaped_feats.view(B, C, -1)  # [B, C, N*H*W]

        # 截断到最近的平方数
        reshaped_feats = reshaped_feats[:, :, :closest_square]  # [B, C, closest_square]

        # 重塑为正方形特征图 [B, C, S, S] where S = closest_side
        enhanced_feats = reshaped_feats.view(B, C, closest_side, closest_side)

        # 金字塔注意力融合
        fused_feats = self.pyramidal_attn(enhanced_feats)  # [B, 128, new_H, new_W]

        # 全局池化并应用特征适配
        img_emb = fused_feats.mean(dim=[2, 3])  # [B, 128]
        img_emb = self.feature_adaptor(img_emb)  # [B, 512]

        return img_emb

    def forward(self, images, texts):
        # 图像编码
        img_emb = self.encode_image(images)

        # 文本编码
        text_emb = self.encode_text(texts)
        text_emb_aug = self.text_augmentation(text_emb)

        # 专家层处理 (移除了因果特征调制)
        img_emb_expert, ortho_loss = self.expert_layer(img_emb)

        # 投影到共享空间
        image_features = self.image_proj(img_emb_expert)
        text_features = self.text_proj(text_emb_aug)

        # 计算对比损失 (移除了因果损失)
        loss, loss_dict = self.contrastive_loss(
            image_features, text_features,
            ortho_loss  # 只保留正交损失
        )
        return loss, loss_dict

    def encode_text(self, texts):
        """多描述文本编码"""
        text_features = []
        for i in range(texts.shape[1]):
            x = self.token_embed(texts[:, i]).type(self.dtype)
            x = x + self.positional_embed.type(self.dtype)
            x = x.permute(1, 0, 2)
            x = self.text_encoder(x)
            x = x.permute(1, 0, 2)
            x = self.ln_final(x).type(self.dtype)
            eos_features = x[torch.arange(x.shape[0]), texts[:, i].argmax(dim=-1)]
            text_features.append(eos_features)
        return torch.stack(text_features, dim=1).mean(dim=1)  # 平均所有描述

    def contrastive_loss(self, img_feats, text_feats, ortho_loss):
        """移除了因果损失的对比损失函数"""
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
        total_loss = (img_loss + txt_loss) + ortho_loss

        return total_loss, {
            "img_loss": img_loss.item(),
            "txt_loss": txt_loss.item(),
            "ortho_loss": ortho_loss.item()
        }

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype


def train_model():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据集参数
    ANNOTATION_PATH = "/home/wuziyi/Project/Flickr/Flickr30K_Entities/data/results_20130124.token"
    IMAGE_DIR = "/home/wuziyi/Project/Flickr/Flickr30K_Entities/data/flickr30k-images"
    NUM_CROPS = 5
    BATCH_SIZE = 16
    EPOCHS = 12
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.05

    # 创建数据集
    print("加载数据集...")
    annotations_df = load_flickr_annotations(ANNOTATION_PATH)

    # 拆分训练集/验证集 (80%/20%)
    image_ids = list(set(annotations_df['image_id']))
    split_idx = int(len(image_ids) * 0.8)
    train_image_ids = image_ids[:split_idx]
    val_image_ids = image_ids[split_idx:]

    # 创建训练集和验证集数据框
    train_annotations = annotations_df[annotations_df['image_id'].isin(train_image_ids)]
    val_annotations = annotations_df[annotations_df['image_id'].isin(val_image_ids)]

    # 训练和验证数据集
    train_dataset = Flickr30kDataset(
        image_dir=IMAGE_DIR,
        annotation_df=train_annotations,
        num_crops=NUM_CROPS
    )

    val_dataset = Flickr30kDataset(
        image_dir=IMAGE_DIR,
        annotation_df=val_annotations,
        num_crops=NUM_CROPS
    )

    # 自定义collate函数处理无效样本和文本填充
    def collate_fn(batch):
        # 过滤无效样本
        batch = [b for b in batch if b is not None and b[0] is not None]

        # 处理图像数据 [B, N_CROPS+1, C, H, W]
        images = torch.stack([item[0] for item in batch])

        # 处理文本数据 [B, 5, 77]
        texts = [item[1] for item in batch]
        # 已经由CLIP tokenizer处理为相同长度77，直接堆叠
        return images, torch.stack(texts)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print(f"训练集大小: {len(train_dataset)} | 验证集大小: {len(val_dataset)}")
    print(f"批大小: {BATCH_SIZE} | 每批步数: {len(train_loader)}")

    # 初始化模型
    clip_model, _ = clip.load("ViT-B/16", device=device, jit=False)
    model = MathEnhancedGOALCLIP(
        clip_model=clip_model,
        device=device,
        num_local_experts=4,
        num_crops=NUM_CROPS
    ).to(device)

    # 初始化权重
    def init_weights(m):
        """特殊参数初始化"""
        if isinstance(m, nn.Linear):
            if 'rff' in m._get_name():
                nn.init.normal_(m.weight, mean=0, std=1 / m.weight.shape[1])
            elif 'expert' in m._get_name():
                nn.init.orthogonal_(m.weight)
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)

    # 设置优化器
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # 学习率调度器 (余弦退火)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * EPOCHS
    )

    # 混合精度训练的梯度缩放器
    scaler = torch.cuda.amp.GradScaler()

    # --------------------- 训练循环 ---------------------
    print("开始训练...")
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        train_steps = 0

        # 训练进度条 - 减少描述更新频率
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"训练 Epoch {epoch + 1}/{EPOCHS}")

        for batch_idx, (images, texts) in train_bar:
            images = images.to(device, non_blocking=True)
            texts = texts.to(device, non_blocking=True)

            # 前向传播 (使用混合精度)
            with torch.cuda.amp.autocast():
                loss, loss_dict = model(images, texts)

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

            # 参数更新
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # 更新学习率
            scheduler.step()

            # 记录损失
            total_train_loss += loss.item()
            train_steps += 1

            # 每50个批次更新一次进度条描述
            if batch_idx % 50 == 0:
                train_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'OrthoLoss': f'{loss_dict.get("ortho_loss", 0.0):.6f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })

        # 计算平均训练损失
        avg_train_loss = total_train_loss / train_steps
        print(f"Epoch {epoch + 1}/{EPOCHS} | 训练损失: {avg_train_loss:.4f}")

        # 验证阶段 (每个epoch结束后执行一次)
        model.eval()
        total_val_loss = 0.0
        val_steps = 0

        # 初始化精度统计指标
        img2txt_top1 = 0.0
        img2txt_top5 = 0.0
        txt2img_top1 = 0.0
        txt2img_top5 = 0.0
        total_samples = 0

        # 验证进度条 - 设置更新频率为10秒
        val_bar = tqdm(val_loader, desc=f"验证 Epoch {epoch + 1}/{EPOCHS}", mininterval=10.0)

        with torch.no_grad():
            for images, texts in val_bar:
                images = images.to(device, non_blocking=True)
                texts = texts.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    loss, loss_dict = model(images, texts)

                    # 获取图像和文本特征
                    img_emb = model.encode_image(images)
                    text_emb = model.encode_text(texts)
                    text_emb_aug = model.text_augmentation(text_emb)

                    # 专家层处理
                    img_emb_expert, _ = model.expert_layer(img_emb)

                    # 投影到共享空间
                    image_features = model.image_proj(img_emb_expert)
                    text_features = model.text_proj(text_emb_aug)

                    # 归一化特征
                    image_features = F.normalize(image_features, dim=-1)
                    text_features = F.normalize(text_features, dim=-1)

                    # 计算相似度矩阵（使用相同的logit_scale）
                    logit_scale = model.logit_scale.exp()
                    sim_matrix = logit_scale * image_features @ text_features.t()

                    # 计算图像到文本和文本到图像的检索精度
                    batch_size = image_features.size(0)
                    labels = torch.arange(batch_size, device=device)

                    # Image-to-Text (i2t) 检索
                    _, i2t_top1_indices = sim_matrix.topk(1, dim=1)
                    _, i2t_top5_indices = sim_matrix.topk(5, dim=1)

                    img2txt_top1 += (i2t_top1_indices.squeeze() == labels).sum().item()
                    img2txt_top5 += (torch.sum(i2t_top5_indices == labels.unsqueeze(1), dim=1)).sum().item()

                    # Text-to-Image (t2i) 检索
                    _, t2i_top1_indices = sim_matrix.topk(1, dim=0)
                    _, t2i_top5_indices = sim_matrix.topk(5, dim=0)

                    txt2img_top1 += (t2i_top1_indices.squeeze() == labels).sum().item()
                    txt2img_top5 += (torch.sum(t2i_top5_indices == labels.unsqueeze(0), dim=0)).sum().item()

                    total_samples += batch_size
                    total_val_loss += loss.item()
                    val_steps += 1

        # 计算平均损失
        avg_val_loss = total_val_loss / val_steps

        # 计算平均精度
        img2txt_top1 /= total_samples
        img2txt_top5 /= total_samples
        txt2img_top1 /= total_samples
        txt2img_top5 /= total_samples

        # 打印详细验证结果
        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"验证损失: {avg_val_loss:.4f} | "
              f"图像->文本 Top-1: {img2txt_top1:.4f} | "
              f"图像->文本 Top-5: {img2txt_top5:.4f} | "
              f"文本->图像 Top-1: {txt2img_top1:.4f} | "
              f"文本->图像 Top-5: {txt2img_top5:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            save_path = f"best_model_epoch{epoch + 1}_loss{avg_val_loss:.4f}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'img2txt_top1': img2txt_top1,
                'img2txt_top5': img2txt_top5,
                'txt2img_top1': txt2img_top1,
                'txt2img_top5': txt2img_top5,
            }, save_path)
            print(f"保存最佳模型到: {save_path}")
            print(f"当前最佳精度: 图像->文本 Top-1: {img2txt_top1:.4f}, 文本->图像 Top-1: {txt2img_top1:.4f}")

    # 保存最终模型
    final_save_path = f"final_model_epochs{EPOCHS}_best{best_epoch + 1}.pth"
    torch.save(model.state_dict(), final_save_path)
    print(f"训练完成! 最终模型保存到: {final_save_path}")


# --------------------- 辅助函数 ---------------------
def set_seed(seed=1):
    """设置所有随机种子确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------- 启动入口 ---------------------
if __name__ == "__main__":
    train_model()
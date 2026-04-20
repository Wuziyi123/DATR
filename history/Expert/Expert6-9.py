import torch
import torch.nn as nn
import numpy as np
import pytorch_wavelets as pw
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
def set_seed(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(0)

# 全局标注字典
annotation_dict = {}


# --------------------- 完全修正的小波-RKHS融合模块 ---------------------
class WaveletRKHSOperator(nn.Module):
    """完全修正的小波-希尔伯特细节增强算子"""

    def __init__(self, in_channels, wavelet='db4', levels=2, sigma=1.0):
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.sigma = sigma
        self.in_channels = in_channels

        # 小波变换对象 (强制使用FP32)
        self.dwt = pw.DWTForward(J=levels, wave=wavelet, mode='zero').to(dtype=torch.float32)

        # 随机傅里叶特征映射
        self.rff_dim = 128
        self.rff_weights = nn.Parameter(torch.randn(in_channels, self.rff_dim // 2,
                                        dtype=torch.float32) * (2 * sigma ** 2))

        # 创建卷积层处理小波系数（强制FP32）
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(in_channels * 3, in_channels, 3, padding=1).to(dtype=torch.float32)
            for _ in range(levels)
        ])

        # 特征降维层（强制FP32）
        self.feature_reducer = nn.Conv2d(levels * in_channels, in_channels, 1).to(dtype=torch.float32)

    def _rff_mapping(self, x):
        """完全修正的随机傅里叶特征映射"""
        B, C, H, W = x.shape
        x_float = x.to(torch.float32)
        # 展平空间维度
        x_flat = x_float.view(B, C, -1)  # [B, C, N]

        # 调整维度进行矩阵乘法
        x_flat = x_flat.permute(0, 2, 1)  # [B, N, C]

        # 矩阵乘法 (确保数据类型一致)
        proj = torch.matmul(x_flat, self.rff_weights)  # [B, N, rff_dim//2]

        # 应用三角函数
        cos_feat = torch.cos(proj)
        sin_feat = torch.sin(proj)
        rff_feat = torch.cat([cos_feat, sin_feat], dim=-1)  # [B, N, rff_dim]

        # 重塑为空间特征图
        rff_feat = rff_feat.permute(0, 2, 1)  # [B, rff_dim, N]
        return rff_feat.view(B, self.rff_dim, H, W).to(x.dtype)

    def _hilbert_transform(self, x):
        """修正的希尔伯特变换，处理数据类型"""
        orig_dtype = x.dtype
        x = x.to(torch.float32)

        # 应用快速傅里叶变换
        f = fftn(x, dim=(-2, -1))

        # 创建希尔伯特滤波器
        n, m = x.size(-2), x.size(-1)
        h = torch.zeros_like(f)
        h[..., :n // 2, :m // 2] = 1
        h[..., -n // 2:, -m // 2:] = 1

        # 应用逆变换
        result = ifftn(f * h, dim=(-2, -1)).real
        return result.to(orig_dtype)

    def dwt_forward(self, x):
        """修改后的DWT前向传播，处理混合精度"""
        orig_dtype = x.dtype
        x_float = x.to(torch.float32)

        # 禁用混合精度，强制使用FP32计算
        yl, yh = self.dwt(x_float)

        # 保持原始精度类型转换
        yl = yl.to(orig_dtype)
        for level in range(len(yh)):
            for i in range(len(yh[level])):
                yh[level][i] = yh[level][i].to(orig_dtype)

        return yl, yh

    def forward(self, x):
        orig_dtype = x.dtype
        B, C, H, W = x.shape
        # 禁用此模块的自动混合精度
        with torch.cuda.amp.autocast(enabled=False):
            x = x.to(torch.float32)
            hilbert_feat = self._hilbert_transform(x).float()
            yl, yh = self.dwt_forward(hilbert_feat)

            wavelet_feats = []
            for level in range(self.levels):
                hl = yh[level][:, :, 0]  # [B, C, H_h, W_h]
                lh = yh[level][:, :, 1]
                hh = yh[level][:, :, 2]

                detail_map = torch.stack([hl, lh, hh], dim=2)  # [B, C, 3, H_h, W_h]
                detail_map = detail_map.reshape(B, C * 3, *detail_map.shape[3:])
                detail_conv = self.wavelet_convs[level](detail_map)
                resized = F.interpolate(detail_conv, size=(H, W),
                                        mode='bilinear', align_corners=False)
                wavelet_feats.append(resized)  # [B, in_channels, H, W]

            concatenated = torch.cat(wavelet_feats, dim=1)  # [B, levels * in_channels, H, W]
            reduced = self.feature_reducer(concatenated)  # [B, in_channels, H, W]
            rff_feat = self._rff_mapping(reduced)

        return rff_feat.to(orig_dtype)


# --------------------- 稳定的金字塔注意力融合 ---------------------
class PyramidalAttentionFusion(nn.Module):
    """简化稳定的金字塔注意力融合模块"""

    def __init__(self, dim):
        super().__init__()
        # 使用多个卷积层提取特征
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, features):
        B, C, H, W = features.shape

        # 应用不同尺度的卷积
        conv1_feat = self.conv1(features)
        conv2_feat = self.conv2(features)

        # 合并特征
        concat_feats = torch.stack([conv1_feat, conv2_feat, features], dim=1)  # [B, 3, C, H, W]

        # 通道注意力
        pooled = concat_feats.mean(dim=[3, 4])  # [B, 3, C]
        channel_weights = self.channel_attn(pooled).view(B, 3, C, 1, 1)

        # 加权融合
        weighted_feats = (concat_feats * channel_weights).mean(dim=1)

        return weighted_feats


# --------------------- 优化专家网络 ---------------------
class OrthogonalExpertLayer(nn.Module):
    """优化的正交约束专家层"""

    def __init__(self, dim, num_experts=4, ortho_weight=0.01):
        super().__init__()
        self.ortho_weight = ortho_weight
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim)
            ).to(torch.float32) for _ in range(num_experts)
        ])

        # 路由网络
        self.router = nn.Sequential(
            nn.Linear(dim, num_experts),
            nn.Softmax(dim=-1)
        ).to(torch.float32)

    def _orthogonal_loss(self, expert_outputs):
        """更稳定的正交约束损失"""
        features = torch.stack(expert_outputs, dim=1)  # [B, K, D]
        # 中心化特征
        features = features - features.mean(dim=0)
        # 计算协方差矩阵
        cov = torch.bmm(features.transpose(1, 2), features) / features.size(0)
        # 单位矩阵
        eye = torch.eye(features.size(2), device=features.device).unsqueeze(0)
        return F.mse_loss(cov, eye)

    def forward(self, x):
        # 动态路由
        x = x.float()  # 强制转换为FP32
        routing_weights = self.router(x)

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x.to(torch.float32)))

        # 正交约束损失
        ortho_loss = self._orthogonal_loss(expert_outputs)

        # 加权融合
        combined = torch.stack(expert_outputs, dim=1) * routing_weights.unsqueeze(2)
        return combined.sum(dim=1), self.ortho_weight * ortho_loss


# --------------------- 文本注意力模块 ---------------------
class TextAttention(nn.Module):
    """文本描述注意力模块"""

    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, text_features):
        """应用注意力加权"""
        # text_features: [B, num_descriptions, dim]
        weights = self.attention(text_features)  # [B, num_descriptions, 1]
        return (text_features * weights).sum(dim=1)  # [B, dim]


# --------------------- 最终稳定版网络架构 ---------------------
class FinalGOALCLIP(nn.Module):
    """完全稳定的多模态对比网络"""

    def __init__(self, clip_model, device, num_local_experts=4, num_crops=5):
        super().__init__()
        # 解冻CLIP部分层
        self.visual = clip_model.visual
        self.text_encoder = clip_model.transformer
        self.token_embed = clip_model.token_embedding
        self.positional_embed = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.num_crops = num_crops
        self.device = device
        self.dwt_levels = 2  # 固定小波分解层数

        # 解冻CLIP后3层
        for name, param in self.visual.named_parameters():
            if "blocks.11" in name or "blocks.10" in name or "blocks.9" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        # 通道降维层
        self.channel_reducer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1)
        )

        # 视觉增强路径
        self.wavelet_rkhs = WaveletRKHSOperator(128)
        self.pyramidal_attn = PyramidalAttentionFusion(128)

        # 特征适配器
        self.feature_adaptor = nn.Sequential(
            nn.Linear(128, 512),
            nn.GELU()
        )

        # 专家层
        self.expert_layer = OrthogonalExpertLayer(512, num_experts=num_local_experts)

        # 文本增强
        self.text_attention = TextAttention(512)
        self.text_augmentation = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Linear(1024, 512)
        )

        # 投影层
        self.image_proj = nn.Linear(512, 256)
        self.text_proj = nn.Linear(512, 256)

        # 学习参数
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 损失参数
        self.alpha = 0.5  # 图文匹配损失权重

    def encode_image(self, images):
        """稳定的图像编码，兼容混合精度"""
        batch = images.shape[0]
        global_img = images[:, 0, ...]
        crop_imgs = images[:, 1:, ...].reshape(-1, *images.shape[-3:])

        # 全局特征处理
        global_feat = self.visual(global_img.type(self.dtype))
        global_feat = global_feat.unsqueeze(1)  # [B, 1, D]

        # 局部特征处理
        crop_feats = self.visual(crop_imgs.type(self.dtype))
        crop_feats = crop_feats.reshape(batch, self.num_crops, -1)

        # 组合全局和局部特征
        all_feats = torch.cat([global_feat, crop_feats], dim=1)  # [B, 1+num_crops, D]
        B, N, D = all_feats.shape

        # 重新排列维度
        spatial_feats = all_feats.permute(0, 2, 1).unsqueeze(-1)  # [B, D, N, 1]

        # 通道降维
        spatial_feats = self.channel_reducer(spatial_feats.to(torch.float32))  # [B, 128, N, 1]

        # 应用小波-RKHS操作
        enhanced_feats = self.wavelet_rkhs(spatial_feats.to(torch.float32))  # [B, 128, N, 1]

        # 金字塔注意力融合
        fused_feats = self.pyramidal_attn(enhanced_feats)  # [B, 128, N, 1]

        # 全局池化并应用特征适配
        img_emb = fused_feats.mean(dim=[2, 3])  # [B, 128]
        img_emb = self.feature_adaptor(img_emb)  # [B, 512]

        return img_emb

    def encode_text(self, texts):
        """增强的文本编码，兼容混合精度"""
        text_features = []
        for i in range(texts.shape[1]):
            # 关键修正：确保文本索引为Long类型
            x = self.token_embed(texts[:, i].long())
            x = x + self.positional_embed
            x = x.permute(1, 0, 2)
            x = self.text_encoder(x)
            x = x.permute(1, 0, 2)
            x = self.ln_final(x)
            # 关键修正：确保argmax结果为Long类型
            eos_features = x[torch.arange(x.shape[0]), texts[:, i].argmax(dim=-1).long()]
            text_features.append(eos_features)

        # 注意力加权融合
        text_tensor = torch.stack(text_features, dim=1)  # [B, num_descriptions, D]
        return self.text_attention(text_tensor)  # [B, D]

    def itm_loss(self, img_feats, text_feats):
        """图文匹配损失"""
        logit_scale = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)
        sim_matrix = logit_scale * img_feats @ text_feats.t()
        itm_labels = torch.eye(img_feats.size(0), device=img_feats.device)
        return F.binary_cross_entropy_with_logits(sim_matrix, itm_labels)

    def forward(self, images, texts):
        # 图像编码
        img_emb = self.encode_image(images)

        # 文本编码
        text_emb = self.encode_text(texts)
        text_emb_aug = self.text_augmentation(text_emb)

        # 专家层处理
        img_emb_expert, ortho_loss = self.expert_layer(img_emb)

        # 投影到共享空间
        image_features = self.image_proj(img_emb_expert)
        text_features = self.text_proj(text_emb_aug)

        # 计算损失
        loss, loss_dict = self.contrastive_loss(image_features, text_features, ortho_loss)
        return loss, loss_dict

    def contrastive_loss(self, img_feats, text_feats, ortho_loss):
        """增强的对比损失"""
        # 归一化特征
        img_feats = F.normalize(img_feats, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)

        # 传统对比损失
        logit_scale = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)
        logits_per_image = logit_scale * img_feats @ text_feats.t()
        logits_per_text = logit_scale * text_feats @ img_feats.t()

        labels = torch.arange(img_feats.size(0), device=img_feats.device)

        img_loss = F.cross_entropy(logits_per_image, labels)
        txt_loss = F.cross_entropy(logits_per_text, labels)

        # ITM损失
        itm_loss = self.itm_loss(img_feats, text_feats)

        # 总损失
        total_loss = (img_loss + txt_loss) + ortho_loss + self.alpha * itm_loss

        return total_loss, {
            "img_loss": img_loss.item(),
            "txt_loss": txt_loss.item(),
            "itm_loss": itm_loss.item(),
            "ortho_loss": ortho_loss.item(),
            "logit_scale": logit_scale.item()
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
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01

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
        batch = [b for b in batch if b is not None and b[0] is not None]
        images = torch.stack([item[0] for item in batch])
        texts = [item[1] for item in batch]
        # 关键修正：确保文本数据保持Long类型
        return images, torch.stack(texts).long()

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
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
    model = FinalGOALCLIP(
        clip_model=clip_model,
        device=device,
        num_local_experts=4,
        num_crops=NUM_CROPS
    ).to(device)

    # 初始化权重
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, WaveletRKHSOperator):
            # 特定初始化
            nn.init.normal_(m.rff_weights, mean=0, std=1 / m.rff_weights.shape[1])

    model.apply(init_weights)

    # 设置优化器
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 混合精度训练的梯度缩放器
    scaler = torch.cuda.amp.GradScaler()

    # --------------------- 训练循环 ---------------------
    print("开始训练...")
    best_top1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        train_steps = 0

        # 训练进度条
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"训练 Epoch {epoch + 1}/{EPOCHS}")

        for batch_idx, (images, texts) in train_bar:
            images = images.to(device, non_blocking=True)
            texts = texts.to(device, non_blocking=True)

            # 前向传播 (使用混合精度)
            with torch.cuda.amp.autocast():
                loss, loss_dict = model(images, texts)

            # 反向传播
            scaler.scale(loss).backward()
            # 在unscale之前显式转换梯度类型
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = param.grad.to(dtype=param.dtype)  # 转换为FP32
            # scaler.unscale_(optimizer)  # 显式转换梯度为FP32
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)

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

        avg_train_loss = total_train_loss / train_steps
        print(f"Epoch {epoch + 1}/{EPOCHS} | 训练损失: {avg_train_loss:.4f}")

        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        val_steps = 0
        img2txt_top1 = 0.0
        txt2img_top1 = 0.0
        total_samples = 0

        val_bar = tqdm(val_loader, desc=f"验证 Epoch {epoch + 1}/{EPOCHS}")

        with torch.no_grad():
            for images, texts in val_bar:
                images = images.to(device)
                texts = texts.to(device)

                with torch.cuda.amp.autocast():
                    loss, _ = model(images, texts)

                    # 编码图像和文本
                    img_emb = model.encode_image(images)
                    text_emb = model.encode_text(texts)

                    # 投影到共享空间
                    img_proj = F.normalize(model.image_proj(img_emb), dim=-1)
                    text_proj = F.normalize(model.text_proj(text_emb), dim=-1)

                    # 计算相似度
                    logit_scale = model.logit_scale.exp()
                    sim_matrix = logit_scale * img_proj @ text_proj.t()

                    batch_size = img_proj.size(0)
                    labels = torch.arange(batch_size, device=device)

                    # Image-to-Text
                    _, i2t_top1 = sim_matrix.topk(1, dim=1)
                    img2txt_top1 += (i2t_top1.squeeze() == labels).sum().item()

                    # Text-to-Image
                    _, t2i_top1 = sim_matrix.topk(1, dim=0)
                    txt2img_top1 += (t2i_top1.squeeze() == labels).sum().item()

                    total_samples += batch_size
                    total_val_loss += loss.item()
                    val_steps += 1

        avg_val_loss = total_val_loss / val_steps
        img2txt_top1 /= total_samples
        txt2img_top1 /= total_samples
        avg_top1 = (img2txt_top1 + txt2img_top1) / 2

        print(f"Epoch {epoch + 1}/{EPOCHS} | 验证损失: {avg_val_loss:.4f} | "
              f"图像->文本 Top-1: {img2txt_top1:.4f} | "
              f"文本->图像 Top-1: {txt2img_top1:.4f} | "
              f"平均Top-1: {avg_top1:.4f}")

        # 保存最佳模型
        if avg_top1 > best_top1:
            best_top1 = avg_top1
            save_path = f"best_model_epoch{epoch + 1}_top1{avg_top1:.4f}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'avg_top1': avg_top1
            }, save_path)
            print(f"保存最佳模型到: {save_path}")

    # 保存最终模型
    final_save_path = f"final_model_epochs{EPOCHS}_top1{best_top1:.4f}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': EPOCHS,
        'avg_top1': best_top1
    }, final_save_path)
    print(f"训练完成! 最终模型保存到: {final_save_path} | 最佳Top-1: {best_top1:.4f}")


if __name__ == "__main__":
    train_model()
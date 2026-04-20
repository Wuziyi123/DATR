import torch
import torch.nn as nn
import numpy as np
import pywt
import scipy.special as sp
from torch.fft import fftn, ifftn
import math
import json
import os
import pickle
import fire
import torch
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
set_seed(42)

# 全局标注字典
annotation_dict = {}


# --------------------- 小波-RKHS融合模块 ---------------------
class WaveletRKHSOperator(nn.Module):
    """小波-希尔伯特细节增强算子 (泛函分析应用)"""

    def __init__(self, in_channels, wavelet='db4', levels=3, sigma=1.0, alpha=0.8):
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
        self.sigma = sigma
        self.alpha = alpha

        # 随机傅里叶特征 (RFF) 映射
        self.rff_dim = 128
        self.rff_weights = nn.Parameter(torch.randn(in_channels, self.rff_dim // 2) * (2 * sigma ** 2))

        # 多尺度小波卷积
        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(4, in_channels, 3, padding=1)
            for _ in range(levels)
        ])

        # 分数阶积分核 (Riemann-Liouville定义)
        self.register_buffer('frac_int_kernel', self._create_frac_int_kernel(alpha))

    def _create_frac_int_kernel(self, alpha):
        """构造分数阶积分核"""
        n = 7  # 核尺寸
        kernel = torch.zeros(n)
        gamma = sp.gamma(alpha)
        for k in range(n):
            t = (k / n) ** (alpha - 1)
            kernel[k] = t / gamma
        return kernel.view(1, 1, -1).float()

    def _rff_mapping(self, x):
        """随机傅里叶特征映射"""
        # 高斯核逼近: k(x,y)=exp(-||x-y||^2/(2σ^2))
        B, C, H, W = x.shape
        proj = torch.matmul(x.view(B, C, -1).permute(0, 2, 1), self.rff_weights)
        # 傅里叶特征构造
        cos_feat = torch.cos(proj)
        sin_feat = torch.sin(proj)
        return torch.cat([cos_feat, sin_feat], dim=-1).view(B, -1, H, W)

    def _frac_integral(self, x):
        """分数阶积分计算"""
        # 水平方向积分
        x_pad = F.pad(x, (3, 3, 0, 0), mode='reflect')
        # 应用分数阶积分核
        return F.conv1d(
            x_pad.view(x.size(0) * x.size(1), 1, -1),
            self.frac_int_kernel.repeat(x.size(1), 1, 1),
            groups=x.size(1),
            padding=0
        ).view_as(x)

    def _hilbert_transform(self, x):
        """希尔伯特变换 (增强边缘特征)"""
        f = fftn(x, dim=(-2, -1))
        h = torch.zeros_like(f)
        h[..., :x.size(-2) // 2, :x.size(-1) // 2] = 1
        h[..., -x.size(-2) // 2:, -x.size(-1) // 2:] = 1
        return ifftn(f * h, dim=(-2, -1)).real

    def forward(self, x):
        B, C, H, W = x.shape
        wavelet_feats = []

        # 希尔伯特变换增强纹理
        hilbert_feat = self._hilbert_transform(x)

        # 多级小波分解
        for level in range(1, self.levels + 1):
            # 小波分解
            coeffs = pywt.wavedec2(hilbert_feat.detach().cpu().numpy(), self.wavelet, level=level)
            # 重构高频细节
            cH, cV, cD = coeffs[1][0], coeffs[1][1], coeffs[1][2]
            detail_feat = np.stack((cH, cV, cD), axis=1)
            detail_tensor = torch.tensor(detail_feat, device=x.device).float()

            # 小波卷积增强
            detail_conv = self.wavelet_convs[level - 1](detail_tensor)
            wavelet_feats.append(F.interpolate(detail_conv, (H, W)))

        # RKHS映射
        rff_feat = self._rff_mapping(torch.cat(wavelet_feats, dim=1).mean(dim=1, keepdim=True))

        # 分数阶积分增强连续性
        integrated_feat = self._frac_integral(x)

        # 特征融合
        combined = integrated_feat + rff_feat
        return combined


# --------------------- 因果推断模块 ---------------------
class FunctionalCausalModule(nn.Module):
    """泛函因果推断模块 (结构化因果模型)"""

    def __init__(self, dim, lambda_reg=0.1):
        super().__init__()
        self.dim = dim
        self.lambda_reg = lambda_reg

        # 因果图权重矩阵 (DAG邻接矩阵)
        self.W = nn.Parameter(torch.zeros(dim, dim))
        self._initialize_weights()

        # Sobolev空间投影
        self.sobolev_proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        # 变分因果编码器
        self.variational_encoder = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU()
        )
        self.variational_decoder = nn.Sequential(
            nn.Linear(dim // 2, dim),
            nn.GELU()
        )

    def _initialize_weights(self):
        """初始化因果图权重"""
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                self.W.data[j, i] = 0.5  # 上三角初始化为零的对称矩阵
                self.W.data[i, j] = 0.5

    def _sobolev_regularization(self, x):
        """Sobolev空间正则化 (拉普拉斯算子)"""
        # 计算二阶导数 (模拟拉普拉斯算子)
        dx = torch.diff(x, dim=1, n=2)
        # Sobolev范数: R(f) = ∫||∇f||^2 dx
        return torch.mean(dx ** 2)

    def _h_function(self):
        """NOTEARS约束函数 (确保无环图)"""
        # h(W) = tr(e^{W*W}) - dim
        return torch.trace(torch.matrix_exp(self.W * self.W)) - self.dim

    def forward(self, global_feat, local_feats):
        """变分因果推理"""
        # 结构化因果建模
        causal_input = torch.cat([global_feat] + local_feats, dim=1)
        causal_graph = torch.matmul(causal_input, self.W)  # X = W^T Z

        # Sobolev空间投影
        sobolev_proj = self.sobolev_proj(causal_graph)

        # 变分因果编码
        z_mean = self.variational_encoder(sobolev_proj)
        z_std = torch.exp(z_mean)  # 简化方差估计
        z = z_mean + z_std * torch.randn_like(z_std)

        # 解码反事实表示
        counterfactual = self.variational_decoder(z)

        # 计算正则化损失
        reg_loss = self.lambda_reg * self._sobolev_regularization(counterfactual)

        # NOTEARS约束
        h_val = self._h_function()

        return counterfactual, reg_loss, h_val


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
    """正交约束专家层 (因果感知)"""

    def __init__(self, dim, num_experts=4):
        super().__init__()
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

        # 因果调制模块
        self.causal_modulation = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def _orthogonal_loss(self, expert_outputs):
        """专家正交约束损失"""
        # 计算专家特征的相关性
        features = torch.stack(expert_outputs, dim=1)  # [B, K, D]
        cov = torch.matmul(features.permute(0, 2, 1), features)  # [B, D, D]
        identity = torch.eye(features.size(2), device=features.device).unsqueeze(0)
        return F.mse_loss(cov, identity.expand_as(cov)) * 0.1

    def forward(self, x, causal_feat):
        """正交特征融合 (带因果调制)"""
        # 动态路由
        routing_weights = self.router(x)

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
            weight = routing_weights[:, i].unsqueeze(-1)
            weighted_outs.append(out * weight)
        combined_expert = sum(weighted_outs)

        # 因果特征调制
        modulation = self.causal_modulation(causal_feat).unsqueeze(-1)
        modulated_out = combined_expert * modulation

        return modulated_out, ortho_loss


# --------------------- 完整网络架构 ---------------------
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
        for param in self.visual.parameters():
            param.requires_grad_(False)
        for param in self.text_encoder.parameters():
            param.requires_grad_(False)
        for param in [self.token_embed, self.positional_embed, self.ln_final, self.text_projection]:
            param.requires_grad_(False)

        # 增强的视觉处理路径
        self.wavelet_rkhs = WaveletRKHSOperator(512)
        self.pyramidal_attn = PyramidalAttentionFusion(512)
        self.causal_module = FunctionalCausalModule(512)
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
        crop_feats = self.visual(crop_imgs.type(self.dtype))
        crop_feats = crop_feats.reshape(batch, self.num_crops, -1)

        # 小波-RKHS增强
        all_feats = torch.cat([global_feat, crop_feats], dim=1)
        B, N, D = all_feats.shape
        H = W = int(D ** 0.5)
        spatial_feats = all_feats.view(B, N, H, W)

        # 应用小波-RKHS操作
        enhanced_feats = torch.cat([
            self.wavelet_rkhs(spatial_feats[:, i:i + 1]) for i in range(N)
        ], dim=1)

        # 金字塔注意力融合
        fused_feats = self.pyramidal_attn(enhanced_feats)
        img_emb = fused_feats.mean(dim=[2, 3])  # 全局池化
        return img_emb, all_feats

    def forward(self, images, texts):
        # 图像编码
        img_emb, img_features = self.encode_image(images)

        # 文本编码
        text_emb = self.encode_text(texts)
        text_emb_aug = self.text_augmentation(text_emb)

        # 因果推理
        global_feat = img_features[:, 0].unsqueeze(1)
        local_feats = [img_features[:, i].unsqueeze(1) for i in range(1, img_features.size(1))]
        causal_feat, reg_loss, h_val = self.causal_module(global_feat, local_feats)

        # 专家层处理 (带正交约束)
        img_emb_expert, ortho_loss = self.expert_layer(img_emb, causal_feat)

        # 投影到共享空间
        image_features = self.image_proj(img_emb_expert)
        text_features = self.text_proj(text_emb_aug)

        # 计算对比损失
        loss, loss_dict = self.contrastive_loss(
            image_features, text_features,
            reg_loss + ortho_loss + h_val
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

    def contrastive_loss(self, img_feats, text_feats, causal_loss):
        """因果增强对比损失"""
        # 归一化特征
        img_feats = F.normalize(img_feats, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)

        # 计算相似度
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * img_feats @ text_feats.t()
        logits_per_text = logit_scale * text_feats @ img_feats.t()

        # 对比目标
        labels = torch.arange(img_feats.size(0), device=img_feats.device)

        # 对比损失
        img_loss = F.cross_entropy(logits_per_image, labels)
        txt_loss = F.cross_entropy(logits_per_text, labels)

        # 总损失
        total_loss = (img_loss + txt_loss) + causal_loss

        return total_loss, {
            "img_loss": img_loss.item(),
            "txt_loss": txt_loss.item(),
            "causal_loss": causal_loss.item()
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
    EPOCHS = 20
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
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print(f"训练集大小: {len(train_dataset)} | 验证集大小: {len(val_dataset)}")
    print(f"批大小: {BATCH_SIZE} | 每批步数: {len(train_loader)}")

    # 初始化模型
    clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
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

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        train_steps = 0

        for batch_idx, (images, texts) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")):
            # 移动到设备
            images = images.to(device, non_blocking=True)
            texts = texts.to(device, non_blocking=True)

            # 前向传播 (使用混合精度)
            with torch.cuda.amp.autocast():
                loss, loss_dict = model(images, texts)

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度裁剪
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

            # 每50个批次打印一次状态
            if batch_idx % 50 == 0:
                lr = optimizer.param_groups[0]['lr']
                img_loss = loss_dict.get('img_loss', 0.0)
                txt_loss = loss_dict.get('txt_loss', 0.0)
                causal_loss = loss_dict.get('causal_loss', 0.0)

                print(f"Epoch {epoch + 1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | ImgLoss: {img_loss:.4f} | "
                      f"TxtLoss: {txt_loss:.4f} | CausalLoss: {causal_loss:.4f} | LR: {lr:.2e}")

        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for images, texts in tqdm(val_loader, desc=f"验证 {epoch + 1}/{EPOCHS}"):
                images = images.to(device, non_blocking=True)
                texts = texts.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    loss, _ = model(images, texts)

                total_val_loss += loss.item()
                val_steps += 1

        # 计算平均损失
        avg_train_loss = total_train_loss / train_steps
        avg_val_loss = total_val_loss / val_steps

        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f"best_model_epoch{epoch + 1}_loss{avg_val_loss:.4f}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, save_path)
            print(f"保存最佳模型到: {save_path}")

    # 保存最终模型
    final_save_path = f"final_model_epochs{EPOCHS}.pth"
    torch.save(model.state_dict(), final_save_path)
    print(f"训练完成! 最终模型保存到: {final_save_path}")


# --------------------- 辅助函数 ---------------------
def set_seed(seed=42):
    """设置所有随机种子确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------- 启动入口 ---------------------
if __name__ == "__main__":
    train_model()


# --------------------- 辅助函数 ---------------------
def init_weights(m):
    """特殊参数初始化"""
    if isinstance(m, nn.Linear):
        if 'rff' in m._get_name():
            # 随机傅里叶特征初始化 (符合高斯分布)
            nn.init.normal_(m.weight, mean=0, std=1 / m.weight.shape[1])
        elif 'expert' in m._get_name():
            # 专家网络正交初始化
            nn.init.orthogonal_(m.weight)
        else:
            # 标准初始化
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gumbel
import torch.fft
import pywt
import numpy as np
from transformers import BertModel, BertConfig, BertTokenizer, AutoModel
from timm.models.vision_transformer import Block
from torch.utils.data import DataLoader
import torch.optim as optim
from Flickr30k_RPN import Flickr30kDataset, load_flickr_annotations
from tqdm import tqdm
import random
import os
import json
from PIL import Image
import torchvision.transforms as transforms

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class MultiPositiveContrastiveLoss(nn.Module):
    def __init__(self, margin=0, max_violation=False):
        super(MultiPositiveContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        # 计算对角线元素（正样本对的相似度）
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # 文本检索损失：每个图像应与其对应文本最相似
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # 图像检索损失：每个文本应与其对应图像最相似
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # 清除对角线（自身比较）
        mask = torch.eye(scores.size(0)) > .5
        I = mask.to(scores.device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # 选择最大违反项或求和
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class MultiPositiveContrastiveLoss(nn.Module):
    """
    1. 可学习的边界参数(margin)和温度系数(temperature)
    2. Top-K负样本策略替代最难的负样本
    3. 加权正样本损失
    4. 自适应损失平衡
    Args:
        init_margin: 初始边界值，默认为0.2
        init_temp: 初始温度系数，默认为0.07
        top_k: 用于负样本选择的top-k值，默认为10
        learnable_params: 是否使边界和温度可学习，默认为True
    """
    def __init__(self, init_margin=0.2, init_temp=0.5, top_k=1, learnable_params=True):
        super().__init__()

        # 可学习的边界参数
        if learnable_params:
            # self.margin = nn.Parameter(torch.tensor(init_margin))
            self.margin = init_margin
            self.temp = nn.Parameter(torch.tensor(init_temp))
        else:
            self.margin = init_margin
            self.temp = init_temp

        self.top_k = max(1, top_k)
        self.learnable_params = learnable_params

        # 自适应权重参数
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 用于平衡两个方向的损失

    def compute_similarity(self, outputs, logit_scale):
        """计算相似度矩阵 (N, 5N)"""
        global_vis = outputs["global_vis"]  # (N, D)
        text_feats = outputs["text_feats"]  # (N, 5, D)
        # 重塑文本特征 (N, 5, D) -> (5N, D)
        text_feats = text_feats.view(-1, text_feats.size(-1))
        # 计算相似度矩阵 (N, 5N)
        similarity = logit_scale * global_vis @ text_feats.t()
        return similarity

    def safe_topk(self, tensor, k, dim=1):
        # 获取操作维度的实际长度
        dim_size = tensor.size(dim)
        # 动态调整k值不超过维度大小
        safe_k = min(k, dim_size)
        # 当k=0时返回空张量
        if safe_k <= 0:
            # 创建与原始形状兼容的零张量
            shape = list(tensor.shape)
            shape[dim] = 1  # 因为我们只需要一个值
            return torch.zeros(shape, device=tensor.device), None
        return torch.topk(tensor, safe_k, dim=dim)

    def forward(self, outputs, logit_scale):
        """
        Args:
            scores: 相似度矩阵，形状为(batch_size, 5 * batch_size)
        Returns:
            损失值
        """
        scores = self.compute_similarity(outputs, logit_scale)
        device = scores.device
        N = scores.size(0)  # 批次大小
        # ===== 应用温度缩放 =====
        # 温度系数控制相似度分布的尖锐程度
        # scores = scores / self.temp
        # ===== 创建正样本掩码 =====
        # 每个图像对应5个文本描述作为正样本
        pos_mask = torch.zeros((N, 5 * N), device=device)
        for i in range(N):
            pos_mask[i, 5 * i:5 * i + 5] = 1

        # ===== 图像查询文本损失 (Image-to-Text) =====
        # 3.1 提取正样本分数 (N, 5)
        pos_scores_i2t = scores[pos_mask.bool()].view(N, 5)

        # 创建负样本掩码 (排除正样本)
        neg_mask_i2t = ~pos_mask.bool()

        # 使用Top-K负样本策略 (N, top_k)
        # 选择最难的top_k个负样本，但避免只使用最难的负样本
        safe_k_i2t = min(self.top_k, 5 * N - 5)
        # 使用安全的topk操作
        topk_values, _ = self.safe_topk(scores * neg_mask_i2t, safe_k_i2t, dim=1)
        neg_scores_i2t = topk_values.mean(dim=1, keepdim=True)  # (N, 1)

        # 计算加权正样本损失
        # 根据正样本分数分配权重 - 分数越高权重越小
        pos_weights = F.softmax(-pos_scores_i2t, dim=1)
        weighted_pos = (pos_weights * pos_scores_i2t).sum(dim=1, keepdim=True)

        # 计算铰链损失
        # 使用边界确保正样本分数高于负样本分数
        cost_i2t = (self.margin + neg_scores_i2t - weighted_pos).clamp(min=0)
        cost_i2t = cost_i2t.mean()

        # ===== 文本查询图像损失 (Text-to-Image) =====
        # 转置相似度矩阵 (5N, N)
        text_scores = scores.t()

        # 创建正样本掩码
        text_pos_mask = torch.zeros((5 * N, N), device=device)
        for j in range(5 * N):
            i = j // 5
            text_pos_mask[j, i] = 1

        # 提取正样本分数 (5N,)
        text_pos_scores = text_scores[text_pos_mask.bool()]

        # 创建负样本掩码
        text_neg_mask = ~text_pos_mask.bool()

        # 使用Top-K负样本策略 (5N, top_k),负样本最大数量为 N - 1
        safe_k_t2i = min(self.top_k, N - 1)
        topk_values, _ = self.safe_topk(text_scores * text_neg_mask, safe_k_t2i, dim=1)
        text_neg_scores = topk_values.mean(dim=1, keepdim=True)  # (5N, 1)

        # 计算铰链损失
        cost_t2i = (self.margin + text_neg_scores - text_pos_scores.unsqueeze(1)).clamp(min=0)
        cost_t2i = cost_t2i.mean()

        # ===== 自适应损失平衡 =====
        # 使用可学习参数平衡两个方向的损失
        i2t_weight = torch.sigmoid(self.alpha)
        t2i_weight = 1 - i2t_weight

        total_loss = i2t_weight * cost_i2t + t2i_weight * cost_t2i
        # total_loss = cost_i2t + cost_t2i

        # ===== 调试信息 =====
        if self.training and torch.isnan(total_loss).any():
            print(f"Warning: NaN loss detected! margin: {self.margin}, temp: {self.temp}")
            print(f"i2t_cost: {cost_i2t}, t2i_cost: {cost_t2i}")

        return total_loss


# ============= 评估函数 =============
def i2t(npts, sims, return_ranks=False):
    """
    计算图像到文本的检索精度
    Args:
        npts: 图像数量
        sims: 相似度矩阵 (N, 5N)
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]  # 按相似度降序排列

        # 找到对应5个文本的位置
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # 计算指标
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(npts, sims, return_ranks=False):
    """
    计算文本到图像的检索精度
    Args:
        npts: 图像数量
        sims: 相似度矩阵 (N, 5N)
    """
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    sims = sims.T  # 转置矩阵 (5N, N)

    for index in range(npts):
        for i in range(5):
            # 当前文本索引
            text_index = 5 * index + i
            inds = np.argsort(sims[text_index])[::-1]  # 按相似度降序排列
            ranks[text_index] = np.where(inds == index)[0][0]
            top1[text_index] = inds[0]

    # 计算指标
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


class MultiContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, outputs):
        image_embeds = outputs["global_vis"]  # (N, D)
        text_embeds = outputs["text_feats"]  # (N, 5, D)
        batch_size = image_embeds.size(0)
        num_texts = text_embeds.size(1)

        # 展平文本特征 [batch_size * num_texts, embed_dim]
        flat_text_embeds = text_embeds.view(batch_size * num_texts, -1)

        # ===== 计算相似度矩阵 =====
        # 图像到文本的相似度 [batch_size, batch_size * num_texts]
        logits_per_image = torch.matmul(
            image_embeds,
            flat_text_embeds.t()
        )

        # 文本到图像的相似度 [batch_size * num_texts, batch_size]
        logits_per_text = torch.matmul(
            flat_text_embeds,
            image_embeds.t()
        )

        # ===== 创建正确的标签 =====
        # 图像到文本的标签：每个图像的正确文本索引
        i2t_labels = torch.arange(
            batch_size,
            device=image_embeds.device
        )

        # 文本到图像的标签：每个文本对应的图像索引
        t2i_labels = torch.arange(
            batch_size,
            device=image_embeds.device
        ).repeat_interleave(num_texts)

        # ===== 计算损失 =====
        # 图像到文本的损失
        loss_i2t = self.cross_entropy(logits_per_image, i2t_labels)

        # 文本到图像的损失
        loss_t2i = self.cross_entropy(logits_per_text, t2i_labels)

        return (loss_i2t + loss_t2i) / 2


def validate(model, val_loader, device):
    """在验证集上评估模型性能"""
    model.eval()
    all_img_feats = []
    all_text_feats = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['images'].to(device)
            batch_size = images.size(0)
            texts = batch['input_ids'].view(batch_size, 5, 77).to(device)
            attention_mask = batch['attention_mask'].view(batch_size, 5, 77).to(device)

            # 前向传播
            outputs = model(images, texts, attention_mask)

            # 保存特征
            img_feats = outputs['global_vis'].cpu().numpy()
            text_feats = outputs['text_feats'].cpu().numpy()

            all_img_feats.append(img_feats)
            all_text_feats.append(text_feats)

        # 合并所有特征
        img_feats = np.concatenate(all_img_feats, axis=0)  # (N, D)
        text_feats = np.concatenate(all_text_feats, axis=0)  # (N, 5, D)

        # 重塑文本特征 (N, 5, D) -> (5N, D)
        text_feats = text_feats.reshape(-1, text_feats.shape[-1])

        # 计算相似度矩阵
        img_feats = torch.tensor(img_feats).to(device)
        text_feats = torch.tensor(text_feats).to(device)

        # logit_scale = model.logit_scale.exp()
        sims = model.logit_scale * img_feats @ text_feats.T  # (N, 5N)
        sims = sims.detach().cpu().numpy()

        # 计算评估指标
        npts = img_feats.size(0)
        r_i2t, r_t2i = i2t(npts, sims), t2i(npts, sims)

        # 提取指标
        r1_i2t, r5_i2t, r10_i2t = r_i2t[0], r_i2t[1], r_i2t[2]
        r1_t2i, r5_t2i, r10_t2i = r_t2i[0], r_t2i[1], r_t2i[2]

        # 计算rsum
        rsum = r1_i2t + r5_i2t + r10_i2t + r1_t2i + r5_t2i + r10_t2i

    return (r1_i2t, r5_i2t, r10_i2t), (r1_t2i, r5_t2i, r10_t2i), rsum


class TextTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_layers=3):
        """
        文本Transformer处理模块
        参数:
        - d_model: 特征维度 (默认512)
        - nhead: 注意力头数 (默认8)
        - num_layers: Transformer层数 (默认3)
        """
        super(TextTransformer, self).__init__()

        # 创建3层Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # FFN隐藏层维度
            dropout=0.1,
            activation='gelu',
            batch_first=True  # 使用(B, S, C)格式
        )
        self.transformer = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        # 可选的CLS位置编码（增强第一个位置的重要性）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0, std=0.02)

    def forward(self, x):
        """
        前向传播
        参数:
        x: 输入特征 [B, 5, 77, 512]
        返回:
        全局文本特征 [B*5, 512]
        """
        B, N, S, D = x.shape  # [batch, 5, 序列长77, 特征512]
        # 1. 维度重组: [B, 5, 77, 512] -> [B*5, 77, 512]
        x = x.view(B * N, S, D)
        # 2. 添加可学习的CLS标记 (可选)
        cls_tokens = self.cls_token.expand(B * N, -1, -1)  # [B*5, 1, 512]
        x = torch.cat([cls_tokens, x], dim=1)  # 现在序列长度=78
        # 3. 通过Transformer编码器
        x = self.transformer(x)  # 输出 [B*5, 78, 512]
        # 4. 提取第一个位置(CLS)作为全局表示
        global_feats = x[:, 0, :]  # [B*5, 512]
        return global_feats


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
        text_feats: [B, 5, M, D] 文本特征
        返回: 增强的视觉特征和正则化损失
        """
        B, N, D = visual_feats.shape
        num_texts = text_feats.size(1)  # 新增：获取文本数量（5）

        # === 多文本聚合（关键步骤） ===
        aggregated_text = torch.mean(text_feats, dim=1)  # [B, M, D]

        # 扩展语义标记
        semantic_tokens = self.semantic_tokens.expand(B, -1, -1)

        # 跨模态注意力：文本引导的语义增强
        attn_output, _ = self.cross_attn(
            query=semantic_tokens,
            key=aggregated_text,
            value=aggregated_text
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


class VisualExpert(nn.Module):
    """CogVLM视觉专家模块：在FFN层注入视觉信息"""

    def __init__(self, embed_dim):
        super().__init__()
        self.vis_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.text_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, text_feats, vis_feats):
        """输入: text_feats [B, L, D], vis_feats [B, D]"""
        vis_expanded = self.vis_proj(vis_feats).unsqueeze(1)  # [B, 1, D]
        text_transformed = self.text_proj(text_feats)  # [B, L, D]
        return text_transformed + vis_expanded


class VarianceControlModule(nn.Module):
    """方差控制模块：减少特征表示的方差"""

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim

        # 特征空间增强
        self.augment_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # 对比对齐变换器
        self.contrastive_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )

    def forward(self, feats):
        """
        feats: [B, N, D] 输入特征
        返回: 增强特征和对比损失
        """
        # 特征空间增强
        aug_feats = self.augment_net(feats)

        # 对比对齐
        combined_feats = torch.cat([feats, aug_feats], dim=1)  # [B, 2N, D]
        transformed_feats = self.contrastive_transformer(combined_feats)

        # 分割特征
        orig_transformed = transformed_feats[:, :feats.size(1)]
        aug_transformed = transformed_feats[:, feats.size(1):]

        # 对比损失
        orig_norm = F.normalize(orig_transformed, dim=-1)
        aug_norm = F.normalize(aug_transformed, dim=-1)
        similarity = torch.bmm(orig_norm, aug_norm.transpose(1, 2))  # [B, N, N]

        # 正样本对角线，负样本非对角线
        pos_sim = torch.diagonal(similarity, dim1=1, dim2=2).mean()
        neg_sim = (similarity.sum(dim=(1, 2)) - pos_sim * feats.size(1)) / (feats.size(1) * (feats.size(1) - 1))
        contrast_loss = 1 - (pos_sim - neg_sim)

        return aug_transformed, contrast_loss


# ===== 特征增强模块 =====

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
        # with torch.no_grad():
        orig_clip_feat = self.clip.encode_image(global_img)

        # 傅里叶增强
        # fourier_img = self.fourier(global_img)

        # 低频特征提取
        # low_freq_feat = self.low_freq_enhancer(fourier_img).flatten(1)

        # 傅里叶增强的CLIP特征
        # with torch.no_grad():
        #     fourier_clip_feat = self.clip.encode_image(fourier_img)
        #
        # # 特征融合
        # fused_feat = torch.cat([
        #     self.clip_enhancer(orig_clip_feat.float()),
        #     self.clip_enhancer(fourier_clip_feat.float()),
        #     low_freq_feat
        # ], dim=1)
        #
        return orig_clip_feat
        # return nn.Linear(fused_feat.size(1), self.embed_dim, device=fused_feat.device)(fused_feat)


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
            nn.Conv2d(embed_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.change_channel = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
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
        # 维度变换
        scene_feats = self.change_channel(scene_feats)
        scene_feats = scene_feats.permute(0, 2, 1).view(B, -1, 7, 7)
        # 空间注意力加权
        attn_weights = self.spatial_attn(scene_feats)
        enhanced_feats = attn_weights * scene_feats

        # 全局池化
        return torch.mean(enhanced_feats, dim=[2, 3])


class TextEncoder(nn.Module):
    """文本编码器：加载预训练双语BERT + 语义增强"""

    def __init__(self, embed_dim=768, model_name="bert-base-uncased"):  # 使用中英双语预训练模型
        super().__init__()
        # 加载预训练模型而非随机初始化
        self.bert = AutoModel.from_pretrained(
            "./my_bert",  # 替换为实际路径
            local_files_only=True  # 强制离线加载
        )
        # 冻结BERT所有参数（关键修改）
        for param in self.bert.parameters():
            param.requires_grad = False
        # 解冻最后3层
        for layer in self.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        # 语义增强模块（兼容BERT输出维度）
        embed_dim = self.bert.config.hidden_size  # 动态获取预训练模型的隐藏层维度
        self.semantic_enhancer = nn.Sequential(
            nn.Linear(embed_dim, 512),  # 线性变换适配
            nn.ReLU(),  # 激活函数与BERT一致
            nn.Linear(512, 512),  # 线性变换适配
        )

    def forward(self, texts, attention_mask):
        B, num_texts, seq_len = texts.shape
        # 展平批次和文本数量维度
        flat_texts = texts.view(B * num_texts, seq_len)
        flat_mask = attention_mask.view(B * num_texts, seq_len)
        # 添加token_type_ids（双语任务需区分语言）
        token_type_ids = torch.zeros_like(flat_texts)  # 单句任务默认全0

        # BERT前向传播
        outputs = self.bert(
            input_ids=flat_texts,
            attention_mask=flat_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        # 使用[CLS]标记作为句子表示
        text_feats = outputs.last_hidden_state[:, 0, :]  # [B*num_texts, seq_len, D]
        enhanced = self.semantic_enhancer(text_feats)
        return enhanced.view(B, num_texts, -1)
        # # 语义增强（逐文本处理）
        # text_feats = text_feats.view(B, num_texts, seq_len, -1)  # [B, num_texts, seq_len, D]
        # enhanced_texts = []
        # for i in range(num_texts):
        #     # 对每个文本的序列特征增强
        #     single_text = text_feats[:, i]  # [B, seq_len, D]
        #     enhanced = self.semantic_enhancer(single_text)  # 增强语义表示
        #     enhanced_texts.append(enhanced)
        #
        # return torch.stack(enhanced_texts, dim=1)  # [B, num_texts, seq_len, D]

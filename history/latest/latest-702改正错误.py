import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gumbel
import torch.fft
import pywt
import numpy as np
from transformers import BertModel, BertConfig, BertTokenizer
from timm.models.vision_transformer import Block
from torch.utils.data import DataLoader
import torch.optim as optim
from latest_utils import VisualExpert, SemanticEmbeddingLearner, VarianceControlModule
from latest_utils import (TextEncoder, ContextEnhancer, LocalWaveletPath,
GlobalFourierPath, TextTransformer, validate)
from latest_datasetloader import Flickr30kDataset, load_flickr_annotations
from tqdm import tqdm
import clip


class LogRobustLoss(nn.Module):
    """基于log的鲁棒损失函数：减少噪声影响"""

    def __init__(self, gamma=1.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        abs_diff = torch.abs(pred - target)
        loss = self.gamma * torch.log(1 + abs_diff / self.gamma)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class EarlyLearningRegularization(nn.Module):
    """早期学习正则化：利用ETP减少噪声影响"""

    def __init__(self, momentum=0.99, alpha=0.5):
        super().__init__()
        self.momentum = momentum
        self.alpha = alpha
        self.register_buffer('ema_pred', None)

    def update(self, current_pred):
        if self.ema_pred is None:
            self.ema_pred = current_pred.detach()
        else:
            self.ema_pred = self.momentum * self.ema_pred + (1 - self.momentum) * current_pred.detach()

    def forward(self, current_pred):
        agreement = torch.sum(self.ema_pred * current_pred, dim=-1)
        return -self.alpha * torch.log(agreement + 1e-8)


# 多正样本对比损失函数
class MultiPositiveContrastiveLoss(nn.Module):
    """
    支持多正样本的对比损失
    Args:
        margin: 正负样本间隔
        max_violation: 是否仅保留最难负样本
    """

    def __init__(self, margin=0.2, max_violation=True):
        super().__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):
        """计算双向对比损失
        Args:
            scores: 相似度矩阵 (N, 5N)
        Returns:
            损失值
        """
        N = scores.size(0)
        device = scores.device

        # 创建正样本掩码 (N, 5N)
        pos_mask = torch.zeros((N, 5 * N), device=device)
        for i in range(N):
            pos_mask[i, 5 * i:5 * i + 5] = 1

        # 文本检索损失（图像作为查询）
        pos_scores = scores[pos_mask.bool()].view(N, 5)
        neg_scores = scores.clone()
        neg_scores[pos_mask.bool()] = -10  # 屏蔽正样本

        if self.max_violation:
            neg_scores = neg_scores.max(dim=1, keepdim=True)[0]  # 最难负样本
        else:
            neg_scores = neg_scores  # 所有负样本

        cost_s = (self.margin + neg_scores - pos_scores).clamp(min=0)
        cost_s = cost_s.mean()

        # 图像检索损失（文本作为查询）
        text_scores = scores.t()  # (5N, N)
        text_pos_mask = torch.zeros((5 * N, N), device=device)
        for j in range(5 * N):
            i = j // 5
            text_pos_mask[j, i] = 1

        text_pos_scores = text_scores[text_pos_mask.bool()]  # (5N,)
        text_neg_scores = text_scores.clone()
        text_neg_scores[text_pos_mask.bool()] = -10  # 屏蔽正样本

        if self.max_violation:
            text_neg_scores = text_neg_scores.max(dim=1, keepdim=True)[0]  # 最难负样本

        cost_im = (self.margin + text_neg_scores - text_pos_scores.unsqueeze(1)).clamp(min=0)
        cost_im = cost_im.mean()

        return cost_s + cost_im


class AdvancedCrossModalRetriever(nn.Module):
    """高级图文检索网络：集成语义嵌入学习和噪声鲁棒机制"""

    def __init__(self, clip_model, num_crops=5, embed_dim=512, num_semantic_tokens=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_crops = num_crops

        # === 全局特征路径：傅里叶增强 ===
        self.global_path = GlobalFourierPath(clip_model, embed_dim)

        # === 局部特征路径：小波增强 ===
        self.local_path = LocalWaveletPath(clip_model, embed_dim, num_crops)

        # === 上下文场景增强模块 ===
        self.context_enhancer = ContextEnhancer(embed_dim)

        # === 文本编码器 ===
        self.text_encoder = TextEncoder(embed_dim)

        # === 语义嵌入学习器 ===
        self.semantic_learner = SemanticEmbeddingLearner(
            embed_dim,
            num_semantic_tokens=num_semantic_tokens
        )

        # === 方差控制模块 ===
        self.variance_control_vis = VarianceControlModule(embed_dim)
        self.variance_control_text = VarianceControlModule(embed_dim)

        # === 早期学习正则化 ===
        self.elr_global = EarlyLearningRegularization()
        self.elr_local = EarlyLearningRegularization()
        self.elr_text = EarlyLearningRegularization()

        # === 图相关推理模块 ===
        self.visual_graph = HierarchicalGraphEncoder(embed_dim, mode="visual")
        self.text_graph = HierarchicalGraphEncoder(embed_dim, mode="text")
        self.cross_graph = CrossGraphReasoner(embed_dim)

        # === 特征净化模块 ===
        self.feature_purifier = CrossModalPurifier(embed_dim)

        # === 多匹配关系建模 ===
        self.multi_match = MultiMatchModule(embed_dim)

        # === 损失模块 ===
        self.loss_module = MultiLevelLoss(embed_dim)
        self.criterion = MultiPositiveContrastiveLoss(margin=0.2, max_violation=True)

    def forward(self, images, texts, attention_mask=None):
        # ===== 特征提取 =====
        # 全局特征 (傅里叶增强)
        global_feat = self.global_path(images[:, 0])

        # 局部特征 (小波增强)
        local_feats = self.local_path(images[:, 1:1 + self.num_crops])

        # 场景上下文增强
        context_feat = self.context_enhancer(global_feat, local_feats)

        # 文本特征
        text_feats = self.text_encoder(texts, attention_mask)

        # ===== 语义嵌入学习 =====
        # 视觉特征语义增强
        B, N, D = local_feats.shape
        enhanced_local, reg_loss = self.semantic_learner(local_feats, text_feats)

        # ===== 方差控制 =====
        # enhanced_local, vc_loss_vis = self.variance_control_vis(enhanced_local)
        # text_feats, vc_loss_text = self.variance_control_text(text_feats)
        # vc_loss = vc_loss_vis + vc_loss_text

        # ===== 特征净化 =====
        puri_vis, puri_text = self.feature_purifier(enhanced_local, text_feats, context_feat)

        # ===== 图相关推理 =====
        vis_graph = self.visual_graph(puri_vis)
        text_graph = self.text_graph(puri_text)
        fused_graph = self.cross_graph(vis_graph, text_graph)

        # ===== 多匹配关系建模 =====  match_scores = [B, num_crops, num_texts, 3]
        match_scores = self.multi_match(context_feat, puri_vis, puri_text, fused_graph)

        # ===== 早期学习正则化更新 =====
        # self.elr_global.update(context_feat)
        # self.elr_local.update(puri_vis.mean(dim=1))
        # self.elr_text.update(puri_text.mean(dim=1))

        return {
            "global_vis": context_feat,
            "local_vis": enhanced_local,
            "text_feats": puri_text,
            "local_feats": puri_vis,
            "text_graph": text_graph,
            "vis_graph": vis_graph,
            "fused_graph": fused_graph,
            "match_scores": match_scores,
            "reg_loss": reg_loss,
            # "vc_loss": vc_loss
        }

    def compute_similarity(self, outputs):
        """计算相似度矩阵 (N, 5N)"""
        global_vis = outputs["global_vis"]  # (N, D)
        text_feats = outputs["text_feats"]  # (N, 5, D)

        # 重塑文本特征 (N, 5, D) -> (5N, D)
        text_feats = text_feats.view(-1, text_feats.size(-1))

        # L2归一化
        global_vis = F.normalize(global_vis, p=2, dim=-1)
        text_feats = F.normalize(text_feats, p=2, dim=-1)

        # 计算相似度矩阵 (N, 5N)
        similarity = global_vis @ text_feats.t()
        return similarity

    def compute_loss(self, outputs):
        """计算多正样本对比损失"""
        similarity = self.compute_similarity(outputs)
        return self.criterion(similarity)

    # def compute_loss(self, outputs, labels):
    #     return self.loss_module(outputs, labels, self)


class CrossModalPurifier(nn.Module):
    def __init__(self, embed_dim, num_experts=4):
        super().__init__()
        self.embed_dim = embed_dim

        # 视觉引导的文本增强模块
        self.visual_experts = nn.ModuleList([
            VisualExpert(embed_dim) for _ in range(num_experts)
        ])
        # 多字幕融合模块
        self.text_semantic_fusion = MultiCaptionFusion(embed_dim)
        # 语义匹配器（增加LayerNorm）
        self.semantic_matcher = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        # 文本引导的视觉去冗余模块
        self.token_importance = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        self.semantic_matcher = nn.Linear(embed_dim, embed_dim)
        self.module_t = TextTransformer()

        # 对齐度与一致度损失 (Law of Vision Representation)
        self.alignment_loss = nn.MSELoss()
        self.correspondence_loss = nn.CosineEmbeddingLoss()

    def visual_guided_text_enhancement(self, global_vis, text_feats):
        """全局特征引导的文本增强"""
        B, num_texts, seq_len, D = text_feats.shape
        text_feats = text_feats.view(B * num_texts, seq_len, D)
        global_vis = global_vis.unsqueeze(1).repeat(1, num_texts, 1).view(B * num_texts, D)

        # 逐层注入视觉专家
        for expert in self.visual_experts:
            text_feats = expert(text_feats, global_vis)

        return text_feats.view(B, num_texts, seq_len, D)

    def text_guided_visual_redundancy_reduction(self, vis_feats, text_feats):
        """基于语义关键点的去冗余 (参考C-Score计算)"""
        # vis_feats = [B, N=局部块数量, 512]
        # text_feats = [B, 5, 77, 512]
        B, N, D = vis_feats.shape
        # 多字幕语义融合
        text_context = self.text_semantic_fusion(text_feats)  # [B, 512]
        # 语义锚点生成
        semantic_anchor = self.semantic_matcher(text_context)  # [B, D]
        # 视觉-文本对齐度计算
        similarity = F.cosine_similarity(
            vis_feats,
            semantic_anchor.unsqueeze(1),  # [B, 1, D]
            dim=-1
        )  # [B, N]

        # Gumbel-Softmax实现可微token选择
        logits = similarity.unsqueeze(-1)  # [B, N, 1]
        gumbel = Gumbel(0, 1).sample(logits.shape).to(logits.device)
        scores = F.softmax((logits + gumbel) / 0.5, dim=1)  # [B, N, 1]

        # 保留TopK重要token
        keep_ratio = 0.7  # 保留70%关键token
        k = int(N * keep_ratio)
        _, topk_indices = torch.topk(scores.squeeze(-1), k, dim=1)

        purified_vis = torch.gather(
            vis_feats,
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, D)
        )
        return purified_vis, topk_indices

    def forward(self, vis_feats, text_feats, global_vis):
        # vis_feats = [B, N=局部块数量, 512] 多个局部快特征
        # text_feats = [B, 5, 77, 512] 文本token特征
        # global_vis = [B, 512] 全局特征
        # 阶段1: 视觉引导文本增强 (使用global_vis)
        enhanced_text = self.visual_guided_text_enhancement(global_vis, text_feats)

        # 阶段2: 文本引导视觉去冗余
        purified_vis, keep_indices = self.text_guided_visual_redundancy_reduction(vis_feats, enhanced_text)

        # 计算视觉表征定律的损失项
        # alignment_loss = self.compute_alignment_loss(global_vis, enhanced_text)
        # correspondence_loss = self.compute_correspondence_loss(purified_vis, keep_indices)
        aggregated_text = self.module_t(enhanced_text)
        _, D = aggregated_text.shape
        aggregated_text = aggregated_text.view(-1, 5, D)
        return purified_vis, aggregated_text

    def compute_alignment_loss(self, global_vis, text_feats):
        """跨模态对齐度损失 (A-Score)"""
        # 全局视觉特征应与文本特征在语义空间对齐
        mean_text = torch.mean(text_feats, dim=[1, 2])  # [B, D]
        return self.alignment_loss(global_vis, mean_text)

    def compute_correspondence_loss(self, vis_feats, keep_indices):
        """视觉一致度损失 (C-Score)"""
        # 保留的token应具备内部一致性
        B, K, D = vis_feats.shape
        anchor = vis_feats[:, 0]  # 以第一个token为锚点
        targets = torch.ones(B).to(vis_feats.device)  # 强制正样本相似
        return self.correspondence_loss(
            anchor.unsqueeze(1).expand(-1, K, -1).reshape(B * K, D),
            vis_feats.reshape(B * K, D),
            targets.unsqueeze(1).expand(-1, K).reshape(-1)
        )


class MultiCaptionFusion(nn.Module):
    """多字幕语义融合：注意力加权聚合"""
    def __init__(self, embed_dim):
        super().__init__()
        self.caption_attn = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.text_pool = nn.Sequential(
            nn.Linear(77 * 512, 512),
            nn.GELU()
        )
    def forward(self, text_feats):
        """输入: [B, 5, 77, 512] 输出: [B, 512]"""
        B, num_caps, seq_len, D = text_feats.shape

        # 字幕级特征提取
        caption_embs = self.text_pool(
            text_feats.reshape(B * num_caps, seq_len * D)
        ).view(B, num_caps, D)  # [B, 5, 512]

        # 注意力加权聚合
        attn_scores = F.softmax(
            self.caption_attn(caption_embs), dim=1
        )  # [B, 5, 1]
        return torch.sum(attn_scores * caption_embs, dim=1)  # [B, 512]


class MultiLevelLoss(nn.Module):
    """多层级损失：集成鲁棒损失和正则化项"""

    def __init__(self, embed_dim, alpha=0.7, beta=0.3, temp_init=0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temp_init))
        self.alpha = alpha  # 鲁棒损失权重
        self.beta = beta  # 正则化权重
        self.log_loss = LogRobustLoss(gamma=1.0)

    def forward(self, outputs, labels, model):
        # 提取特征
        global_vis = outputs["global_feat"]
        text_feats = outputs["text_feats"]
        match_scores = outputs["match_scores"]
        local_feats = outputs["local_feats"]
        B, num_crops, D = local_feats.shape
        num_texts = text_feats.size(1)  # 5个文本

        # === 鲁棒实例级对比损失 ===
        global_text = torch.mean(text_feats, dim=1)
        logits = torch.matmul(global_vis, global_text.t()) / self.temperature

        # 标准交叉熵
        ce_loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)

        # 鲁棒log损失
        robust_loss = self.log_loss(logits, F.one_hot(labels, num_classes=logits.size(-1)).float())

        # 组合损失
        inst_loss = self.alpha * robust_loss + (1 - self.alpha) * ce_loss

        # === 局部对比损失 ===
        local_loss = 0
        for i in range(local_feats.size(1)):
            crop_feats = local_feats[:, i]
            logits = torch.matmul(crop_feats, text_feats.mean(dim=1).t()) / self.temperature

            # 标准交叉熵
            ce_local = F.cross_entropy(logits, labels)

            # 鲁棒log损失
            robust_local = self.log_loss(logits, F.one_hot(labels, num_classes=logits.size(-1)).float())

            local_loss += self.alpha * robust_local + (1 - self.alpha) * ce_local

        local_loss /= local_feats.size(1)

        # === 图一致性损失 ===
        vis_global = outputs["vis_graph"]["graph"]
        text_global = outputs["text_graph"]["graph"]
        graph_loss = F.mse_loss(
            F.normalize(vis_global, dim=-1),
            F.normalize(text_global, dim=-1)
        )

        # === 多匹配损失 ===
        match_loss = 0.0
        match_loss = self.matching_loss(match_scores)

        # === 早期学习正则化 ===
        elr_loss = 0
        # elr_loss += model.elr_global(global_vis).mean()
        # elr_loss += model.elr_local(local_feats.mean(dim=1)).mean()
        # elr_loss += model.elr_text(text_feats.mean(dim=1)).mean()

        # === 正则化项 ===
        # reg_loss = outputs["reg_loss"] + outputs["vc_loss"]
        reg_loss = outputs["reg_loss"]

        # 总损失
        total_loss = (0.4 * inst_loss + 0.3 * local_loss +
                      0.2 * graph_loss + 0.1 * match_loss +
                      self.beta * (elr_loss + reg_loss))

        return total_loss

    def matching_loss(self, match_scores, labels):
        """修改后的多匹配损失：每个局部图像与对应图片的字幕作为正样本，与其他图片的字幕作为负样本"""
        # 输入match_scores维度: [batch_size, num_crops, num_texts, 3]
        # 先合并三种分数：取平均
        mean_scores = torch.mean(match_scores, dim=-1)  # [B, num_crops, num_texts]

        B, num_crops, num_texts = mean_scores.shape
        # 计算每个图片对应的文本数（假设每个图片有相同数量的文本）
        num_texts_per_image = num_texts // B

        # 创建掩码：标记正样本位置 [B, num_texts]
        mask = torch.zeros(B, num_texts, device=mean_scores.device, dtype=torch.bool)
        for i in range(B):
            start_idx = i * num_texts_per_image
            end_idx = start_idx + num_texts_per_image
            mask[i, start_idx:end_idx] = True

        # 扩展掩码到每个局部图像 [B, num_crops, num_texts]
        mask_expanded = mask.unsqueeze(1).expand(-1, num_crops, -1)

        # 计算正样本分数（当前图片的局部图像与对应字幕）
        pos_scores = mean_scores[mask_expanded].view(B, num_crops, -1)  # [B, num_crops, num_texts_per_image]

        # 计算负样本分数（当前图片的局部图像与其他图片的字幕）
        neg_mask = ~mask_expanded
        neg_scores = mean_scores[neg_mask].view(B, num_crops, -1)  # [B, num_crops, num_neg_texts]

        # 计算对比损失
        loss = 0.0
        tau = 0.07  # 温度参数
        for i in range(B):
            for j in range(num_crops):
                # 当前局部图像的正样本分数
                pos = pos_scores[i, j]  # [num_texts_per_image]

                # 当前局部图像的负样本分数
                neg = neg_scores[i, j]  # [num_neg_texts]

                # 计算InfoNCE损失
                numerator = torch.exp(pos / tau).sum()
                denominator = numerator + torch.exp(neg / tau).sum()
                loss += -torch.log(numerator / denominator)

        # 平均损失
        return loss / (B * num_crops)


class HierarchicalGraphEncoder(nn.Module):
    """层级图编码器：节点级+图级"""

    def __init__(self, embed_dim, mode="None"):
        super().__init__()
        self.mode = mode
        # 节点级编码
        self.node_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # 图注意力网络
        self.gat = GraphAttentionNetwork(embed_dim)

        # 图级聚合
        self.graph_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, features):
        if self.mode == "text":
            pass
        # 节点编码
        node_feats = self.node_encoder(features)

        # 图注意力
        attn_feats = self.gat(node_feats)

        # 图级特征
        graph_feat = self.graph_pool(attn_feats.permute(0, 2, 1)).squeeze(-1)
        return {"node": attn_feats, "graph": graph_feat}


class GraphAttentionNetwork(nn.Module):
    """图注意力网络：GATv2实现"""

    def __init__(self, embed_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_dim // heads

        # 线性变换
        self.W = nn.Linear(embed_dim, heads * self.head_dim)
        self.a = nn.Parameter(torch.empty(1, heads, 2 * self.head_dim))

        # 初始化
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x):
        B, N, _ = x.shape

        # 线性变换
        h = self.W(x).view(B, N, self.heads, self.head_dim)

        # 计算注意力系数
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1, -1)
        a_input = torch.cat([h_i, h_j], dim=-1)

        e = torch.einsum('bijkh,nh->bijn', a_input, self.a.squeeze(0))
        e = F.leaky_relu(e, 0.2)

        # 注意力权重
        attention = F.softmax(e, dim=-1)

        # 加权聚合
        output = torch.einsum('bijn,bjnh->binh', attention, h)
        return output.reshape(B, N, -1)


class CrossGraphReasoner(nn.Module):
    """跨图推理器：双向图注意力"""

    def __init__(self, embed_dim):
        super().__init__()
        # 视觉到文本的图注意力
        self.vis_to_text = GraphAttentionNetwork(embed_dim)

        # 文本到视觉的图注意力
        self.text_to_vis = GraphAttentionNetwork(embed_dim)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, vis_graph, text_graph):
        # 节点特征
        vis_nodes = vis_graph["node"]
        text_nodes = text_graph["node"]

        # 视觉→文本
        vis_to_text = self.vis_to_text(
            torch.cat([vis_nodes, text_nodes], dim=1)
        )[:, :vis_nodes.size(1)]

        # 文本→视觉
        text_to_vis = self.text_to_vis(
            torch.cat([text_nodes, vis_nodes], dim=1)
        )[:, :text_nodes.size(1)]

        # 特征融合
        vis_pool = torch.mean(vis_to_text, dim=1)  # [32, 512]
        text_pool = torch.mean(text_to_vis, dim=1)  # [32, 512]
        fused_feats = torch.cat([vis_pool, text_pool], dim=-1)  # [32, 1024]
        # fused_feats = torch.cat([vis_to_text, text_to_vis], dim=-1)
        return self.fusion(fused_feats)


class MultiMatchModule(nn.Module):
    """多匹配关系建模：三路相似度"""

    def __init__(self, embed_dim):
        super().__init__()
        # 全局匹配
        self.global_match = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        # 局部匹配
        self.local_match = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        # 图增强匹配
        self.graph_match = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        # 图特征适配器
        self.graph_adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, global_vis, local_vis, text_feats, fused_graph):
        batch_size, num_crops, D = local_vis.shape
        pooled_text = text_feats
        num_texts = text_feats.size(1)

        # 图特征适配（关键修复）
        # fused_graph: [B, D] -> [B, num_crops, D] (广播到每个crop)
        graph_feat = self.graph_adapter(fused_graph)
        graph_feat = graph_feat.unsqueeze(1).expand(-1, num_crops, -1)  # [B, num_crops, D]

        # 1. 全局匹配分数
        global_text = torch.mean(pooled_text, dim=1)  # [B, D]
        global_score = self.global_match(
            torch.cat([global_vis, global_text], dim=1)
        )  # [B, 1]

        # 2. 局部匹配分数（并行计算）
        # 扩展维度: [B, num_crops, D] -> [B, num_crops, num_texts, D]
        local_expanded = local_vis.unsqueeze(2).expand(-1, -1, num_texts, -1)
        text_expanded = pooled_text.unsqueeze(1).expand(-1, num_crops, -1, -1)

        # 拼接特征并计算分数
        local_pairs = torch.cat([local_expanded, text_expanded], dim=-1)
        local_scores = self.local_match(local_pairs)  # [B, num_crops, num_texts, 1]

        # 3. 图增强匹配（关键修复）
        # 扩展图特征: [B, num_crops, D] -> [B, num_crops, num_texts, D]
        graph_expanded = graph_feat.unsqueeze(2).expand(-1, -1, num_texts, -1)

        # 拼接三重特征
        triplet_feats = torch.cat([local_expanded, text_expanded, graph_expanded], dim=-1)
        graph_scores = self.graph_match(triplet_feats)  # [B, num_crops, num_texts, 1]

        # 组合所有分数（修复维度对齐）
        global_score = global_score.view(batch_size, 1, 1, 1).expand(-1, num_crops, num_texts, -1)
        return torch.cat([global_score, local_scores, graph_scores], dim=-1)  # [B, num_crops, num_texts, 3]


# ===== 训练主函数 =====

def train_retriever():
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载CLIP模型
    clip_model, _ = clip.load("ViT-B/32", device=device)
    for param in clip_model.parameters():
        param.requires_grad = False

    # 初始化数据集
    annotation_path = "/home/wuziyi/Project/Flickr/Flickr30K_Entities/data/results_20130124.token"
    image_dir = "/home/wuziyi/Project/Flickr/Flickr30K_Entities/data/flickr30k-images"
    rpn_file = "/home/wuziyi/Project/Flickr/Flickr30K_Entities/flickr30k_rpn_proposals-U.json"
    annotation_df = load_flickr_annotations(annotation_path)

    # 新划分：1000测试集 + 1000评估集 + 其余训练集
    image_ids = list(set(annotation_df['image_id']))
    random.shuffle(image_ids)
    train_image_ids = image_ids[:200]  # 前1000张测试集
    eval_image_ids = image_ids[200:400]  # 后续1000张评估集
    test_image_ids = image_ids[400:600]  # 其余为训练集
    # 创建训练集和验证集数据框
    train_annotations = annotation_df[annotation_df['image_id'].isin(train_image_ids)]
    eval_annotations = annotation_df[annotation_df['image_id'].isin(eval_image_ids)]
    test_annotations = annotation_df[annotation_df['image_id'].isin(test_image_ids)]

    train_dataset = Flickr30kDataset(image_dir, train_annotations, rpn_proposals_file=rpn_file)
    eval_dataset = Flickr30kDataset(image_dir, eval_annotations)
    test_dataset = Flickr30kDataset(image_dir, test_annotations)
    # 创建数据加载器
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=16,shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    model = AdvancedCrossModalRetriever(clip_model, num_crops=12).to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f}M")

    # 优化器配置
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(dataloader) * 5,
        T_mult=1,
        eta_min=1e-6
    )

    # 训练循环
    num_epochs = 20
    best_epoch = 0
    val_rsum = 0
    best_test_rsum = 0
    best_rsum = 0
    # 记录所有epoch的测试结果
    all_test_results = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        epoch_start = time.time()

        for batch_idx, batch in progress_bar:
            images = batch['images'].to(device)
            batch_size = images.size(0)
            # 获取多文本数据 [B, 5, 77]
            texts = batch['input_ids'].view(batch_size, 5, 77).to(device)
            attention_mask = batch['attention_mask'].view(batch_size, 5, 77).to(device)

            if images is None:
                continue

            # 创建多文本标签 (每张图片有5个匹配文本)
            labels = torch.arange(batch_size).to(device)  # [0, 1, ..., 31]
            # labels = torch.arange(batch_size).repeat_interleave(5).to(device)

            optimizer.zero_grad()

            # 前向传播
            outputs = model(images, texts, attention_mask)

            # 计算损失
            loss = model.compute_loss(outputs)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        train_time = time.time() - epoch_start
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")


        # ===== 验证阶段 =====
        val_start = time.time()
        (r1_i2t, r5_i2t, r10_i2t), (r1_t2i, r5_t2i, r10_t2i), rsum = validate(model, eval_loader, device)
        print(f"\nEpoch {epoch + 1} Validation Results:")
        print(f"Image to Text: R@1={r1_i2t:.1f}, R@5={r5_i2t:.1f}, R@10={r10_i2t:.1f}")
        print(f"Text to Image: R@1={r1_t2i:.1f}, R@5={r5_t2i:.1f}, R@10={r10_t2i:.1f}")
        print(f"RSUM: {rsum:.1f}")
        val_time = time.time() - val_start

        # ===== 测试阶段 =====
        test_start = time.time()
        test_i2t, test_t2i, test_rsum = validate(model, test_loader, device)
        test_time = time.time() - test_start

        # 记录测试结果
        test_results = {
            "epoch": epoch + 1,
            "r1_i2t": test_i2t[0],
            "r5_i2t": test_i2t[1],
            "r10_i2t": test_i2t[2],
            "r1_t2i": test_t2i[0],
            "r5_t2i": test_t2i[1],
            "r10_t2i": test_t2i[2],
            "rsum": test_rsum
        }
        all_test_results.append(test_results)

        # ===== 打印epoch总结 =====
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{num_epochs} Summary:")
        print(f"{'-' * 60}")
        print(f"Training Loss: {avg_loss:.4f} | Time: {train_time:.2f}s")
        print(f"Validation RSUM: {val_rsum:.1f} | Time: {val_time:.2f}s")
        print(f"Test RSUM: {test_rsum:.1f} | Time: {test_time:.2f}s")
        print(f"{'-' * 60}")
        print("Test Results:")
        print(f"Image to Text: R@1 = {test_i2t[0]:.1f}, R@5 = {test_i2t[1]:.1f}, R@10 = {test_i2t[2]:.1f}")
        print(f"Text to Image: R@1 = {test_t2i[0]:.1f}, R@5 = {test_t2i[1]:.1f}, R@10 = {test_t2i[2]:.1f}")
        print(f"{'=' * 60}\n")

        # 保存最佳模型（基于验证集RSUM）
        if val_rsum > best_rsum:
            best_rsum = val_rsum
            best_epoch = epoch + 1
            best_test_rsum = test_rsum
            torch.save(model.state_dict(), "best_advanced_retriever.pth")
            print(
                f"🌟 New best model saved at epoch {epoch + 1} with val RSUM={val_rsum:.1f}, test RSUM={test_rsum:.1f}")

        # 保存当前模型
        torch.save(model.state_dict(), f"retriever_epoch{epoch + 1}.pth")

    print("Training completed!")


if __name__ == "__main__":
    train_retriever()


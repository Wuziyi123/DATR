import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.fft
# import pywt
import numpy as np
from transformers import BertModel, BertConfig, BertTokenizer, AutoModel
from timm.models.vision_transformer import Block
from tqdm import tqdm
import timm
import random
import os
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import SwinModel, AutoImageProcessor
# 在文档2的开头添加
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("FlashAttention not available, using standard attention")


# 添加以下相似度计算函数和类
# latest_util.py
class EncoderSimilarity(nn.Module):
    def __init__(self, embed_size, sim_dim, self_layers=2):
        super(EncoderSimilarity, self).__init__()

        self.sim_eval_w = nn.Linear(2 * sim_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.module_t = nn.ModuleList([SelfReasoning(sim_dim) for i in range(self_layers)])
        self.module_v = nn.ModuleList([SelfReasoning(sim_dim) for i in range(self_layers)])

        self.init_weights()

    def forward(self, sim_emb_t_list, sim_emb_v_list):
        sim_all = []
        n_image = len(sim_emb_v_list)
        n_caption = len(sim_emb_t_list)

        for i in range(len(sim_emb_t_list)):
            sim_emb_t = sim_emb_t_list[i]
            sim_emb_v = sim_emb_v_list[i]
            # compute the final similarity vector

            for module in self.module_t:
                sim_emb_t = module(sim_emb_t)
            sim_vec_t = sim_emb_t[:, 0, :]

            for module in self.module_v:
                sim_emb_v = module(sim_emb_v)
            sim_vec_v = sim_emb_v[:, 0, :]

            # compute the final similarity score
            sim_vec = torch.cat([sim_vec_t, sim_vec_v], dim=-1)

            sim_i = self.sigmoid(self.sim_eval_w(sim_vec))
            sim_all.append(sim_i)

        # (n_image, n_caption)
        sim_all = torch.cat(sim_all, 1)
        return sim_all

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# 添加 SCAN_attention 函数
def SCAN_attention(query, context, smooth=9.0, eps=1e-8):
    """
    query: (batch_size, queryL, dim)
    context: (batch_size, sourceL, dim)
    """
    # 计算注意力分数
    attn = torch.bmm(context, query.permute(0, 2, 1))  # (batch_size, sourceL, queryL)
    attn = F.leaky_relu(attn, 0.1)
    attn = l2norm(attn, dim=2)

    # 应用 softmax 获取注意力权重
    attn = attn.permute(0, 2, 1)  # (batch_size, queryL, sourceL)
    attn = F.softmax(attn * smooth, dim=2)

    # 计算加权上下文
    weighted_context = torch.bmm(attn, context)  # (batch_size, queryL, dim)
    return l2norm(weighted_context, dim=-1)

def l2norm(X, dim, eps=1e-8):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def l2norm_glo(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X, norm


class TextSA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()

        self.embedding_local = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                             nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_global = nn.Sequential(nn.Linear(embed_dim, embed_dim),
                                              nn.Tanh(), nn.Dropout(dropout_rate))
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local words and raw global text
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, L)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final text, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global


# latest_util.py
class SimilarityComputer(nn.Module):
    """计算全局和局部相似度矩阵（改进版）"""
    def __init__(self, embed_dim, sim_dim=256, mode="train", num_self_layers=2):
        super(SimilarityComputer, self).__init__()
        self.sim_dim = sim_dim
        self.mode = mode

        self.t_global_w = TextSA(embed_dim, 0.4)
        self.dropout = nn.Dropout(0.4)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.emb_local_li = nn.Linear(embed_dim, embed_dim)
        self.emb_local_bn = nn.BatchNorm1d(36)
        self.emb_global_li = nn.Linear(embed_dim, embed_dim)
        self.emb_global_bn = nn.BatchNorm1d(embed_dim)
        self.emb_common = nn.Linear(embed_dim, 1)

        # 全局特征投影层
        self.global_proj = nn.Linear(embed_dim, embed_dim)
        # self.sim_enc = EncoderSimilarity(embed_dim, sim_dim)

        self.sim_tranloc_wv = nn.Sequential(
            nn.Linear(embed_dim, sim_dim),
            nn.ReLU(),
        )
        self.sim_tranloc_wt = nn.Sequential(nn.Linear(embed_dim, sim_dim),
            nn.ReLU(),
        )
        self.sim_tranglo_wg = nn.Sequential(
            nn.Linear(embed_dim, sim_dim),
            nn.ReLU(),
        )
        self.relu = nn.ReLU()

        # 初始化权重
        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, attention_mask, global_img, global_text, local_img, local_text):
        """
        改进的相似度计算前向传播：
        0. 根据attention_mask准确计算每个文本的有效长度
        1. 添加L2归一化确保特征稳定性
        2. 优化SCAN注意力计算流程
        3. 修正维度变换逻辑
        """
        n_image = local_img.size(0)
        n_caption = local_text.size(0)
        # 获取有效的文本长度
        text_lengths = attention_mask.sum(dim=-1)

        # get enhanced global images by self-attention
        # img_ave = torch.mean(local_img, 1)
        # l_emb = self.dropout(self.tanh(self.emb_local_bn(self.emb_local_li(local_img))))
        # g_emb = self.dropout(self.tanh(self.emb_global_bn(self.emb_global_li(img_ave))))
        # g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        # common = l_emb.mul(g_emb)
        # weights_raw = self.emb_common(common)
        # weights = self.softmax(weights_raw.squeeze(2)).unsqueeze(2)
        # new_global = (weights * local_img).sum(dim=1)
        # img_glo, norm_glo = l2norm_glo(new_global, dim=-1)

        # === 相似度计算 ===
        sim_t_list = []
        sim_v_list = []
        global_sim_list = []

        for i in range(n_caption):
            n_words = text_lengths[i]  # [B] 当前文本的有效长度
            text_feat = local_text[i, :n_words, :].unsqueeze(0)
            # text_feat = local_text[i].unsqueeze(0)
            text_i_expand = text_feat.repeat(n_image, 1, 1)

            # 双向SCAN注意力  [B, L_text, D]
            text_context = SCAN_attention(text_feat.repeat(n_image, 1, 1), local_img, smooth=9.0)  # [B, M, D]
            sim_loc_t = torch.pow(torch.sub(text_context, text_i_expand), 2)
            sim_loc_t = l2norm(self.sim_tranloc_wt(sim_loc_t), dim=-1)
            sim_t_list.append(sim_loc_t)

            # text_ave_feat = torch.mean(text_feat, 1)
            text_glo_feat = self.t_global_w(text_feat, global_text)
            img_context = SCAN_attention(local_img, text_i_expand, smooth=9.0)  # [B, num_crops, D]
            # 计算相似度矩阵  [B, N_crop, D]
            sim_loc_v = torch.pow(torch.sub(img_context, local_img), 2)
            sim_loc_v = l2norm(self.sim_tranloc_wv(sim_loc_v), dim=-1)

            # === 全局相似度计算 ===
            # cap_glo4par_i = cap_glo_i.repeat(36, 1).unsqueeze(0)
            sim_glo = torch.pow(torch.sub(global_img, text_glo_feat), 2)
            sim_glo = l2norm(self.sim_tranglo_wg(sim_glo), dim=-1)

            global_sim_list.append(sim_glo)
            sim_golo_v = torch.cat([sim_glo.unsqueeze(1), sim_loc_v], 1)
            sim_v_list.append(sim_golo_v)

        return sim_t_list, sim_v_list


class SelfReasoning(nn.Module):
    """自推理模块"""
    def __init__(self, sim_dim):
        super(SelfReasoning, self).__init__()
        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()
        self.init_weights()

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
        sim_self = torch.bmm(sim_edge, sim_emb)
        return self.relu(self.sim_graph_w(sim_self))

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MultiPositiveContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0.2, max_violation=True, alpha=0.5, temperature=0.07):
        super(MultiPositiveContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation
        # self.alpha = alpha
        # self.temperature = temperature
        # self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, outputs):
        # compute image-sentence score matrix
        """使用相似度矩阵计算损失"""
        scores = outputs["sim_matrix"]  # (B, B)
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = mask.to('cuda:0')
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        contrast_loss = cost_s.sum() + cost_im.sum()

        # 添加InfoNCE风格损失
        # logits_per_image = scores / self.temperature
        # logits_per_text = scores.t() / self.temperature
        #
        # labels = torch.arange(scores.size(0), device=scores.device)
        #
        # loss_i2t = self.cross_entropy(logits_per_image, labels)
        # loss_t2i = self.cross_entropy(logits_per_text, labels)
        #
        # nce_loss = (loss_i2t + loss_t2i) / 2
        # # 组合损失
        # total_loss = self.alpha * contrast_loss + (1 - self.alpha) * nce_loss

        return contrast_loss


# class MultiContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super().__init__()
#         self.temperature = temperature
#         self.cross_entropy = nn.CrossEntropyLoss()
#
#     def forward(self, outputs):
#         image_embeds = outputs["global_vis"]  # (N, D)
#         text_embeds = outputs["text_feats"]  # (N, 5, D)
#         batch_size = image_embeds.size(0)
#         num_texts = text_embeds.size(1)
#
#         # 展平文本特征 [batch_size * num_texts, embed_dim]
#         flat_text_embeds = text_embeds.view(batch_size * num_texts, -1)
#
#         # ===== 计算相似度矩阵 =====
#         # 图像到文本的相似度 [batch_size, batch_size * num_texts]
#         logits_per_image = torch.matmul(
#             image_embeds,
#             flat_text_embeds.t()
#         )
#
#         # 文本到图像的相似度 [batch_size * num_texts, batch_size]
#         logits_per_text = torch.matmul(
#             flat_text_embeds,
#             image_embeds.t()
#         )
#
#         # ===== 创建正确的标签 =====
#         # 图像到文本的标签：每个图像的正确文本索引
#         i2t_labels = torch.arange(
#             batch_size,
#             device=image_embeds.device
#         )
#
#         # 文本到图像的标签：每个文本对应的图像索引
#         t2i_labels = torch.arange(
#             batch_size,
#             device=image_embeds.device
#         ).repeat_interleave(num_texts)
#
#         # ===== 计算损失 =====
#         # 图像到文本的损失
#         loss_i2t = self.cross_entropy(logits_per_image, i2t_labels)
#
#         # 文本到图像的损失
#         loss_t2i = self.cross_entropy(logits_per_text, t2i_labels)
#
#         return (loss_i2t + loss_t2i) / 2


class IntraModalMetricConsistencyLoss(nn.Module):
    """模态内度量一致性损失: 约束图像-图像和文本-文本相似度的一致性"""
    def __init__(self, margin=0.2):
        super(IntraModalMetricConsistencyLoss, self).__init__()
        self.margin = margin

    def forward(self, image_features, text_features):
        """
        参数:
            image_features: 图像特征 [B, D]
            text_features: 文本特征 [B, D]
        返回:
            loss: 模态内一致性损失值
        """
        batch_size = image_features.size(0)

        # 计算图像模态内相似度矩阵 S(I_i, I_j)
        image_sim = torch.matmul(image_features, image_features.t())  # [B, B]

        # 计算文本模态内相似度矩阵 S(T_i, T_j)
        text_sim = torch.matmul(text_features, text_features.t())  # [B, B]

        # 计算相似度差异的 L2 范数（ ||S(I_i,I_j) - S(T_j,T_i)||_2^2）
        # 注意：文本相似度矩阵需要转置以匹配 S(T_j, T_i)
        diff = torch.norm(image_sim - text_sim.t(), p=2, dim=1)  # [B,]

        # 应用 hinge loss: [diff - margin]_+
        loss = torch.clamp(diff - self.margin, min=0.0)

        # 归一化：除以 batch_size 的平方（|B_re|^2）
        loss = loss.sum() / (batch_size ** 2)

        return loss


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
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
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
    """全局路径信息"""
    def __init__(self, clip_model, embed_dim=512):
        super().__init__()
        self.clip = clip_model
        self.embed_dim = embed_dim

        # 傅里叶变换
        # self.fourier = FourierEnhancement()

        self.image_global_proj = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, global_img):
        # 原始CLIP特征
        with torch.no_grad():
            orig_clip_feat = self.clip.encode_image(global_img)

        orig_clip_feat = self.image_global_proj(orig_clip_feat)

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
        # self.wavelet = WaveletEnhancement(threshold=0.15)

        # 高频增强模块
        # self.high_freq_enhancer = nn.Sequential(
        #     nn.Conv2d(3, 32, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.AdaptiveMaxPool2d((7, 7))
        # )
        self.image_local_proj = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, crops):
        """crops: [B, num_crops, 3, 224, 224]"""
        batch_size = crops.size(0)
        enhanced_feats = []
        local_feats = []

        for i in range(self.num_crops):
            crop = crops[:, i]

            # 小波增强
            # wave_crop = self.wavelet(crop)

            # 原始CLIP特征
            with torch.no_grad():
                orig_feat = self.clip.encode_image(crop)

            orig_feat = self.image_local_proj(orig_feat)
            local_feats.append(orig_feat)

        return torch.stack(local_feats, dim=1)

        #     # 小波增强的CLIP特征
        #     with torch.no_grad():
        #         wave_feat = self.clip.encode_image(wave_crop)
        #
        #     # 高频特征提取
        #     high_freq_feat = self.high_freq_enhancer(wave_crop).flatten(1)
        #
        #     # 特征融合
        #     fused_feat = torch.cat([
        #         orig_feat.float(),
        #         wave_feat.float(),
        #         high_freq_feat
        #     ], dim=1)
        #
        #     enhanced_feats.append(nn.Linear(fused_feat.size(1), self.embed_dim, device=fused_feat.device)(fused_feat))
        #
        # return torch.stack(enhanced_feats, dim=1)


# from pytorch_wavelets import DWTForward
# class DWT(nn.Module):
#     """ 离散小波变换 (Haar基) """
#     def __init__(self):
#         super().__init__()
#         self.dwt = DWTForward(J=1, wave='haar', mode='zero')
#         self.channel_reduce = nn.Linear(256, 64)
#
#     def forward(self, x):
#         # x: (B, C, H, W)
#         with torch.no_grad():
#             Yl, Yh = self.dwt(x)
#             # 合并低频与高频分量
#             LH, HL, HH = Yh[0][:, :, 0, ...], Yh[0][:, :, 1, ...], Yh[0][:, :, 2, ...]
#             wave =  torch.cat([Yl, LH, HL, HH], dim=1).flatten(2)
#             return self.channel_reduce(wave.permute(0, 2, 1))


class IDWT(nn.Module):
    """ 逆小波变换 """
    def forward(self, coeffs):
        # coeffs: (B, 4C, H, W)
        B, C4, H, W = coeffs.shape
        C = C4 // 4
        coeffs = coeffs.reshape(B * C, 4, H, W)
        recs = []
        for c in coeffs:
            cA, cH, cV, cD = c[0], c[1], c[2], c[3]
            rec = pywt.idwt2((cA, (cH, cV, cD)), 'haar')
            recs.append(rec)
        return torch.tensor(np.array(recs), device=coeffs.device).reshape(B, C, H * 2, W * 2)


class WaveletAttention(nn.Module):
    """ 小波域多头注意力 """

    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dwt = DWT()

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, C//heads)

        # 小波变换K/V (降采样75%)
        H = W = int(math.sqrt(N))
        k_dwt = self.dwt(k.permute(0, 1, 3, 2).reshape(B * self.heads, -1, H, W))
        v_dwt = self.dwt(v.permute(0, 1, 3, 2).reshape(B * self.heads, -1, H, W))

        # 注意力计算
        attn = (q.reshape(B * self.heads, N, -1) @ k_dwt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v_dwt).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class WaveViTBlock(nn.Module):
    """ 小波增强Transformer块 """

    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WaveletAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.energy_filter = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # 动态谱滤波
        # energy = self.energy_filter(x.mean(1))  # (B,1)
        # x = x + energy * self.attn(self.norm1(x))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


import torch.nn.functional as F
from timm.models.layers import trunc_normal_
class ConvScaleDown(nn.Module):
    """卷积下采样模块（替代插值操作）"""
    def __init__(self, in_chans=3, out_chans=768, scale_factor=8):
        super().__init__()
        # 计算卷积核大小和步幅
        kernel_size = scale_factor
        stride = scale_factor

        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size,
                              stride=stride, bias=False)
        # self.bn = nn.BatchNorm2d(out_chans)
        # self.relu = nn.ReLU(inplace=True)  # 启用inplace

    def forward(self, x):
        return self.conv(x)  # 恢复特征增强


from clip.model import Transformer
class MultiScaleTransformerBlock(nn.Module):
    """独立处理单一尺度的完整Transformer块（孪生网络共享此模块）"""
    def __init__(self, dim, num_heads, trans_model=None, depth=12):
        super().__init__()
        # 位置编码和分类token
        scale = dim ** -0.5
        # self.cls_token0 = nn.Parameter(scale * torch.randn(1, 1, dim))
        self.cls_token0 = nn.Parameter(scale * torch.randn(dim))
        self.cls_token1 = nn.Parameter(scale * torch.randn(dim))
        self.cls_token2 = nn.Parameter(scale * torch.randn(dim))
        # self.cls_token1 = trans_model.class_embedding
        # self.pos_embed1 = trans_model.positional_embedding
        # self.cls_token2 = nn.Parameter(scale * torch.randn(1, 1, dim))
        self.pos_embed0 = nn.Parameter(scale * torch.randn(28 ** 2 + 1, dim))
        self.pos_embed1 = nn.Parameter(scale * torch.randn(14 ** 2 + 1, dim))
        self.pos_embed2 = nn.Parameter(scale * torch.randn(7 ** 2 + 1, dim))
        # 28*28+1 14*14+1 7*7+1

        # Transformer层堆叠
        # self.transformer_layers = Transformer(768, depth, num_heads)
        # self.transformer_layers = clip_model.visual.transformer
        self.transformer_layers = nn.ModuleList([
            trans_model.transformer.resblocks[i] for i in range(depth)
        ])
        # for i in range(12):  # 前10层冻结
        #     for param in self.transformer_layers[i].parameters():
        #         param.requires_grad = False
        # for i in range(10, 12):  # 前10层冻结
        #     for param in self.transformer_layers[i].parameters():
        #         param.requires_grad = True

        self.ln_pre = LayerNorm(dim)
        self.norm = LayerNorm(dim)
        # self.proj = trans_model.proj
        self.proj = nn.Parameter(scale * torch.randn(dim, 512))

        self.init_weights()

    def init_weights(self):
        # for m in self.children():
        #     if isinstance(m, nn.Linear):
        #         r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
        #         m.weight.data.uniform_(-r, r)
        #         m.bias.data.fill_(0)
        #     elif isinstance(m, nn.BatchNorm1d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

        # 初始化权重
        trunc_normal_(self.cls_token0, std=.02)
        trunc_normal_(self.cls_token1, std=.02)
        trunc_normal_(self.cls_token2, std=.02)
        trunc_normal_(self.pos_embed0, std=.02)
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)

    def forward(self, x):
        # 输入x: [B, C, H, W]
        B, C, H, W = x.shape

        # 展平空间维度 [B, C, H, W] -> [B, L, C]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # 添加分类token和位置编码
        if H == 28:
            # cls_tokens = self.cls_token1.expand(B, -1, -1).to(x.dtype)
            x = torch.cat([self.cls_token0.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                     dtype=x.dtype, device=x.device), x], dim=1)
            x = x + self.pos_embed0.to(x.dtype)
        elif H == 14:
            x = torch.cat([self.cls_token1.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                     dtype=x.dtype, device=x.device), x], dim=1)
            # x = torch.cat([cls_tokens, x], dim=1)
            x = x + self.pos_embed1.to(x.dtype)
        else:
            x = torch.cat([self.cls_token2.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                     dtype=x.dtype, device=x.device), x], dim=1)
            x = x + self.pos_embed2.to(x.dtype)

        # 后2层：微调（保留梯度）
        # for i in range(10, 12):
        #     x = self.transformer_layers[i](x)

        x = self.ln_pre(x).permute(1, 0, 2)  # NLD -> LND

        with torch.no_grad():
            for i in range(12):
                x = self.transformer_layers[i](x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.norm(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        # 返回分类token特征 [B, D]
        return x


class AdaptiveFusion(nn.Module):
    """自适应多尺度特征融合模块（改进版）"""
    def __init__(self, dim, num_scales):
        super().__init__()
        # 通道注意力权重生成
        self.channel_attn = nn.Sequential(
            nn.Linear(dim * num_scales, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, num_scales),
            nn.Softmax(dim=-1)
        )
        self.init_weights()

    def init_weights(self):  # 统一Kaiming初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)

    def forward(self, features):
        # 堆叠特征 [B, num_scales, D]
        stacked = torch.stack(features, dim=1)
        # 拼接所有尺度的特征 [B, S*D]
        flattened = stacked.flatten(start_dim=1)  # [B, S*D]
        # 生成注意力权重 [B, S]
        attn_weights = self.channel_attn(flattened)  # [B, S]
        # 加权求和 [B, S, D] * [B, S, 1] -> [B, S, D] -> 求和 [B, D]
        fused = torch.sum(stacked * attn_weights.unsqueeze(-1), dim=1)
        return fused


class LocalPath(nn.Module):
    """多尺度Transformer集成网络（孪生权重共享+卷积下采样）"""
    def __init__(self, dim=768, heads=8, trans_model=None, scales=[28, 14, 7,]):
        super().__init__()
        # 卷积下采样器（生成多尺度输入）
        scale = dim ** -0.5
        # self.conv1 = trans_model.conv1
        self.downsamplers = nn.ModuleList([
            ConvScaleDown(scale_factor=8),  # 224->28
            ConvScaleDown(scale_factor=16),  # 224->14
            ConvScaleDown(scale_factor=32)  # 224->7
        ])

        # 共享权重的Transformer块（孪生网络核心）
        self.shared_transformer = MultiScaleTransformerBlock(dim, heads, trans_model)

        # 特征融合模块
        self.fusion = AdaptiveFusion(512, len(scales))

        # 最终投影层
        # self.proj = nn.Parameter(scale * torch.randn(dim, 512))
        self.image_local_proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 512),
        )
        # self.proj = nn.Linear(dim, 512)
        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, crops):
        """
        crops: 输入图像块 [B, num_crops, 3, 224, 224]
        返回: 多尺度融合特征 [B, num_crops, 512]
        """
        B, N, _, H, W = crops.shape
        all_crop_features = []

        for i in range(N):
            crop = crops[:, i]  # [B, 3, 224, 224]
            scale_features = []

            # 生成多尺度输入（卷积下采样）
            scales = [
                self.downsamplers[0](crop),  # [B, 3, 28, 28]
                self.downsamplers[1](crop),  # [B, 3, 14, 14]
                self.downsamplers[2](crop)  # [B, 3, 7, 7]
            ]

            # 各尺度独立通过共享Transformer
            for x in scales:
                scale_features.append(self.shared_transformer(x))

            # 自适应融合
            # fused_feat = self.shared_transformer(x)
            # fused_feat = self.fusion(scale_features) @ self.proj
            # fused_feat = feat @ self.proj

            fused_feat = self.fusion(scale_features)
            fused_feat = self.image_local_proj(fused_feat)
            all_crop_features.append(fused_feat)
            # all_crop_features.append(fused_feat @ self.proj)

        return torch.stack(all_crop_features, dim=1)  # [B, N, 512]


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
        self.embed_dim = embed_dim
        self.bert = AutoModel.from_pretrained(
            "./my_bert",  # 替换为实际路径
            local_files_only=True  # 强制离线加载
        )
        # 冻结BERT所有参数（关键修改）
        for param in self.bert.parameters():
            param.requires_grad = False
        # 解冻最后3层
        for layer in self.bert.encoder.layer[-3:]:
            for param in layer.parameters():
                param.requires_grad = True

        # 语义增强模块（兼容BERT输出维度）
        hidden_dim = self.bert.config.hidden_size  # 动态获取预训练模型的隐藏层维度
        # 添加投影层统一维度为512
        self.pro_lo = nn.Sequential(
            nn.Linear(768, embed_dim),
            nn.BatchNorm1d(77),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        self.pro_go = nn.Sequential(
            nn.Linear(768, self.embed_dim),
            nn.BatchNorm1d(self.embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim, embed_dim)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

        last_hidden_state = outputs.last_hidden_state  # (B*num_texts, seq_len, hidden_dim)
        # 根据attention mask过滤[PAD] token的特征
        # 将mask扩展为与hidden_state相同的维度，用于元素级乘法
        mask_expanded = flat_mask.unsqueeze(-1).expand_as(last_hidden_state).float()
        # 应用mask，将[PAD] token的特征置为零
        masked_hidden_state = last_hidden_state * mask_expanded

        # 提取全局文本特征 ([CLS] token)
        global_text_feats = masked_hidden_state[:, 0, :]  # (B*num_texts, 768)
        # 提取局部文本特征 (所有 tokens)
        local_text_feats = masked_hidden_state  # (B*num_texts, seq_len, 768)

        # 投影到512维
        global_text_feats = self.pro_go(global_text_feats)  # (B*num_texts, 512)
        local_text_feats = self.pro_lo(local_text_feats)  # (B*num_texts, seq_len, 512)

        # 重塑维度 (B, num_texts=1, 512) 和 (B, num_texts=1, seq_len, 512)
        global_text_feats = global_text_feats.view(B, num_texts, -1)
        local_text_feats = local_text_feats.view(B, num_texts, seq_len, -1)

        return global_text_feats, local_text_feats


def set_seed(seed):
    print(f"Setting seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 新增的净化模块
# class TextGuidedImagePurification(nn.Module):
#     """文本引导的图像净化模块"""
#     def __init__(self, embed_dim, num_heads=8):
#         super().__init__()
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim, num_heads, batch_first=True
#         )
#         # self.norm = nn.LayerNorm(embed_dim)
#         self.gate = nn.Sequential(
#             nn.Linear(embed_dim * 2, embed_dim),
#             nn.BatchNorm1d(32),
#             nn.Sigmoid()
#         )
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.children():
#             if isinstance(m, nn.Linear):
#                 r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
#                 m.weight.data.uniform_(-r, r)
#                 m.bias.data.fill_(0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def forward(self, image_feats, text_feats):
#         # image_feats: [B, N, D]
#         # text_feats: [B, M, D]
#         attn_output, _ = self.cross_attn(
#             query=image_feats,
#             key=text_feats,
#             value=text_feats
#         )
#
#         # 门控融合
#         gate_input = torch.cat([image_feats, attn_output], dim=-1)
#         gate_val = self.gate(gate_input)
#         purified = gate_val * image_feats + (1 - gate_val) * attn_output
#
#         return purified
#
#
# class ImageGuidedTextPurification(nn.Module):
#     """图像引导的文本净化模块"""
#     def __init__(self, embed_dim, num_heads=8):
#         super().__init__()
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim, num_heads, batch_first=True
#         )
#         # self.norm = nn.LayerNorm(embed_dim)
#         self.gate = nn.Sequential(
#             nn.Linear(embed_dim * 2, embed_dim),
#             nn.BatchNorm1d(77),
#             nn.Sigmoid()
#         )
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.children():
#             if isinstance(m, nn.Linear):
#                 r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
#                 m.weight.data.uniform_(-r, r)
#                 m.bias.data.fill_(0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def forward(self, text_feats, image_feats):
#         # text_feats: [B, M, D]
#         # image_feats: [B, N, D]
#         attn_output, _ = self.cross_attn(
#             query=text_feats,
#             key=image_feats,
#             value=image_feats
#         )
#
#         # 门控融合
#         gate_input = torch.cat([text_feats, attn_output], dim=-1)
#         gate_val = self.gate(gate_input)
#         purified = gate_val * text_feats + (1 - gate_val) * attn_output
#
#         return purified


# 新增的token削减模块
class AdaptiveTokenPruning(nn.Module):
    """自适应token削减模块"""
    def __init__(self, embed_dim, keep_ratio=0.7, min_tokens=5):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens
        self.importance_proj = nn.Linear(embed_dim, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feats, guidance_feats=None):
        # feats: [B, N, D]
        # guidance_feats: [B, M, D] (可选，用于引导削减)

        B, N, D = feats.shape
        keep_num = max(self.min_tokens, int(N * self.keep_ratio))

        # 计算重要性分数
        if guidance_feats is not None:
            # 使用引导特征计算相关性
            similarity = torch.bmm(feats, guidance_feats.transpose(1, 2))  # [B, N, M]
            importance_scores = similarity.mean(dim=2)  # [B, N]
        else:
            # 使用自身特征计算重要性
            importance_scores = self.importance_proj(feats).squeeze(-1)  # [B, N]

        # 选择top-k tokens
        _, keep_indices = torch.topk(importance_scores, keep_num, dim=1)
        pruned_feats = torch.gather(
            feats, 1,
            keep_indices.unsqueeze(-1).expand(-1, -1, D)
        )

        return pruned_feats, keep_indices


# 新增的分布校准模块
class DistributionCalibration(nn.Module):
    """预分布校准模块"""
    def __init__(self, embed_dim, modality_specific=True):
        super().__init__()
        self.modality_specific = modality_specific
        self.embed_dim = embed_dim

        if modality_specific:
            self.image_calibration = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                # nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
            self.text_calibration = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                # nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )
        else:
            self.calibration = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                # nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, embed_dim)
            )

    def forward(self, feats, modality='image'):
        # 保存原始形状
        original_shape = feats.shape

        # 展平除最后一个维度外的所有维度
        if len(feats.shape) > 2:
            feats = feats.reshape(-1, self.embed_dim)

        # 应用校准
        if self.modality_specific:
            if modality == 'image':
                calibrated = self.image_calibration(feats)
            else:
                calibrated = self.text_calibration(feats)
        else:
            calibrated = self.calibration(feats)

        # 恢复原始形状
        if len(original_shape) > 2:
            calibrated = calibrated.reshape(original_shape)

        return calibrated


class TextGuidedImagePurification(nn.Module):
    """稳定的文本引导图像净化模块"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 可学习的门控机制
        self.gate = nn.Parameter(torch.tensor(0.1))

        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 投影层（使用更小的初始化）
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 丢弃层
        self.dropout = nn.Dropout(dropout)

        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(self.head_dim ** 0.5))

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # Xavier均匀初始化
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1e-2)
        nn.init.xavier_uniform_(self.kv_proj.weight, gain=1e-2)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1e-2)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.kv_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image_feats, text_feats):
        """
        稳定的文本引导图像净化
        输入: image_feats [B, N, D], text_feats [B, M, D]
        输出: 净化后的图像特征 [B, N, D]
        """
        residual = image_feats
        B, N, D = image_feats.shape
        M = text_feats.shape[1]

        # 归一化输入
        image_feats = self.norm1(image_feats)
        text_feats = self.norm2(text_feats)

        # 查询投影 (图像特征)
        q = self.q_proj(image_feats).reshape(B, N, self.num_heads, self.head_dim)

        # 键值投影 (文本特征)
        kv = self.kv_proj(text_feats).reshape(B, M, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]  # [B, M, num_heads, head_dim]

        # 调整维度
        q = q.transpose(1, 2)  # [B, num_heads, N, head_dim]
        k = k.transpose(1, 2).transpose(2, 3)  # [B, num_heads, head_dim, M]
        v = v.transpose(1, 2)  # [B, num_heads, M, head_dim]

        # 计算注意力分数
        attn_scores = torch.matmul(q, k) / self.temperature  # [B, num_heads, N, M]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力
        attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, N, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(B, N, D)  # [B, N, D]

        # 输出投影
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # 门控残差连接（逐步引入净化）
        output = residual + self.gate * attn_output

        return output


class ImageGuidedTextPurification(nn.Module):
    """稳定的图像引导文本净化模块"""

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 可学习的门控机制
        self.gate = nn.Parameter(torch.tensor(0.1))

        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 丢弃层
        self.dropout = nn.Dropout(dropout)

        # 温度参数
        self.temperature = nn.Parameter(torch.tensor(self.head_dim ** 0.5))

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1e-2)
        nn.init.xavier_uniform_(self.kv_proj.weight, gain=1e-2)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1e-2)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.kv_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, text_feats, image_feats):
        """
        稳定的图像引导文本净化
        输入: text_feats [B, M, D], image_feats [B, N, D]
        输出: 净化后的文本特征 [B, M, D]
        """
        residual = text_feats
        B, M, D = text_feats.shape
        N = image_feats.shape[1]

        # 归一化输入
        text_feats = self.norm1(text_feats)
        image_feats = self.norm2(image_feats)

        # 查询投影 (文本特征)
        q = self.q_proj(text_feats).reshape(B, M, self.num_heads, self.head_dim)

        # 键值投影 (图像特征)
        kv = self.kv_proj(image_feats).reshape(B, N, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]  # [B, N, num_heads, head_dim]

        # 调整维度
        q = q.transpose(1, 2)  # [B, num_heads, M, head_dim]
        k = k.transpose(1, 2).transpose(2, 3)  # [B, num_heads, head_dim, N]
        v = v.transpose(1, 2)  # [B, num_heads, N, head_dim]

        # 计算注意力分数
        attn_scores = torch.matmul(q, k) / self.temperature  # [B, num_heads, M, N]
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力
        attn_output = torch.matmul(attn_weights, v)  # [B, num_heads, M, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(B, M, D)  # [B, M, D]

        # 输出投影
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # 门控残差连接
        output = residual + self.gate * attn_output

        return output


class GradualPurificationScheduler:
    """渐进式净化调度器"""

    def __init__(self, total_epochs, warmup_epochs=10):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1

    def get_gate_value(self):
        """根据训练进度返回门控值"""
        if self.current_epoch < self.warmup_epochs:
            # 热身阶段：逐步增加净化强度
            return 0.1 * (self.current_epoch / self.warmup_epochs)
        else:
            # 正常训练阶段：全强度净化
            return 0.1


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torch.autograd import Function
import numpy as np
class GaussianDistributionAdapter(nn.Module):
    """
    高斯分布适配器：为特征估计高斯参数（均值和方差）
    Gaussian Mixture Cross-Modal Alignment
    """
    def __init__(self, embed_dim, latent_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        # 均值估计网络
        self.mean_estimator = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, embed_dim)
        )

        # 方差估计网络（对数方差，确保正值）
        self.logvar_estimator = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, embed_dim)
        )

        # 分布变换网络
        self.distribution_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """
        输入: [B, ..., D]
        输出: 校准后的特征 [B, ..., D], 分布损失
        """
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.reshape(-1, self.embed_dim)

        # 估计高斯参数
        mean = self.mean_estimator(x)
        logvar = self.logvar_estimator(x)
        std = torch.exp(0.5 * logvar)

        # 重参数化采样
        eps = torch.randn_like(std)
        z = mean + eps * std

        # 应用分布变换
        z_transformed = self.distribution_proj(z)

        if len(original_shape) > 2:
            z_transformed = z_transformed.reshape(original_shape)

        # 返回校准后的特征和分布参数（用于损失计算）
        return z_transformed, mean, logvar


class MutualInformationEstimator(nn.Module):
    """
    互信息估计器：使用InfoNCE和Jensen-Shannon估计器
    引用：ICML 2024 "Cross-Modal Mutual Information Maximization"
    """

    def __init__(self, embed_dim, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def info_nce_loss(self, visual_feats, text_feats):
        """InfoNCE互信息估计"""
        batch_size = visual_feats.size(0)

        # 投影到相同空间
        v = self.projector(visual_feats)
        t = self.projector(text_feats)

        # 归一化
        v = F.normalize(v, p=2, dim=1)
        t = F.normalize(t, p=2, dim=1)

        # 计算相似度矩阵
        logits = torch.matmul(v, t.t()) / self.temperature

        # 创建标签
        labels = torch.arange(batch_size, device=v.device)

        # 对称InfoNCE损失
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)

        return (loss_i2t + loss_t2i) / 2

    def jsd_mi_estimator(self, visual_feats, text_feats):
        """
        Jensen-Shannon散度互信息估计器
        更稳定且偏差较小
        """
        # 联合分布样本
        joint_pairs = torch.cat([visual_feats, text_feats], dim=1)

        # 边缘分布样本（打乱配对）
        shuffled_idx = torch.randperm(text_feats.size(0))
        marginal_pairs = torch.cat([visual_feats, text_feats[shuffled_idx]], dim=1)

        # 使用判别器区分联合分布和边缘分布
        joint_score = self.discriminator(joint_pairs)
        marginal_score = self.discriminator(marginal_pairs)

        # JSD互信息估计
        mi_estimate = F.softplus(joint_score) - F.softplus(marginal_score)
        return -mi_mean()  # 最大化互信息


class AdvancedDistributionCalibration(nn.Module):
    """
    高级分布校准模块：集成高斯适配和互信息最大化
    """
    def __init__(self, embed_dim, mode='both', use_mi=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.mode = mode  # 'image', 'text', or 'both'
        self.use_mi = use_mi

        # 模态特定的分布适配器
        if mode in ['image', 'both']:
            self.image_adapter = GaussianDistributionAdapter(embed_dim)
        if mode in ['text', 'both']:
            self.text_adapter = GaussianDistributionAdapter(embed_dim)

        # 互信息估计器
        if use_mi:
            self.mi_estimator = MutualInformationEstimator(embed_dim)

        # 共享投影层
        self.shared_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim, embed_dim),
        )
        # self.shared_projection = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim * 2),
        #     nn.LayerNorm(embed_dim * 2),
        #     nn.GELU(),
        #     nn.Linear(embed_dim * 2, embed_dim),
        #     nn.LayerNorm(embed_dim)
        # )
        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, image_feats, text_feats=None):
        """
        输入:
        - image_feats: 图像特征 [B, D] 或 [B, N, D]
        - text_feats: 文本特征 [B, D] 或 [B, M, D]

        输出: 校准后的特征，分布校准损失
        """
        calibration_loss = 0.0
        mi_loss = 0.0

        # 图像分布校准
        if self.mode in ['image', 'both']:
            image_feats_cal, mean_i, logvar_i = self.image_adapter(image_feats)
            # KL散度正则化（防止方差崩溃）
            kl_loss_i = -0.5 * torch.sum(1 + logvar_i - mean_i.pow(2) - logvar_i.exp())
            calibration_loss += kl_loss_i / image_feats.size(0)
        else:
            image_feats_cal = image_feats

        # 文本分布校准
        if text_feats is not None and self.mode in ['text', 'both']:
            text_feats_cal, mean_t, logvar_t = self.text_adapter(text_feats)
            kl_loss_t = -0.5 * torch.sum(1 + logvar_t - mean_t.pow(2) - logvar_t.exp())
            calibration_loss += kl_loss_t / text_feats.size(0)
        else:
            text_feats_cal = text_feats

        # 互信息最大化
        if self.use_mi and text_feats is not None:
            # 展平特征以计算MI
            if len(image_feats_cal.shape) > 2:
                image_flat = image_feats_cal.mean(dim=1)  # 平均池化
            else:
                image_flat = image_feats_cal

            if len(text_feats_cal.shape) > 2:
                text_flat = text_feats_cal.mean(dim=1)  # 平均池化
            else:
                text_flat = text_feats_cal

            mi_loss = self.mi_estimator.info_nce_loss(image_flat, text_flat)

        # 共享空间投影
        image_feats_cal = self.shared_projection(image_feats_cal)
        if text_feats_cal is not None:
            text_feats_cal = self.shared_projection(text_feats_cal)

        return image_feats_cal, text_feats_cal, calibration_loss + mi_loss


from transformers import SwinModel, SwinConfig
# class SwinGlobalPath(nn.Module):
#     """Swin-Transformer全局路径（全冻结）"""
#     def __init__(self, embed_dim=512):
#         super().__init__()
#         # 加载预训练的Swin-Transformer模型
#         self.swin_model = SwinModel.from_pretrained("./swin_weights")
#         # self.swin_model = timm.create_model('swin_base_patch4_window7_224',
#         #                                     pretrained=True,
#         #                                     num_classes=0)  # 不包含分类头
#
#         # 冻结所有参数
#         for param in self.swin_model.parameters():
#             param.requires_grad = False
#
#         # Swin输出的特征维度是1024，需要投影到目标维度
#         self.projection = nn.Sequential(
#             nn.Linear(1024, embed_dim),
#             nn.BatchNorm1d(embed_dim),
#             nn.ReLU(),
#             nn.Dropout(p=0.1),
#             nn.Linear(embed_dim, embed_dim)
#         )
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.projection.children():
#             if isinstance(m, nn.Linear):
#                 r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
#                 m.weight.data.uniform_(-r, r)
#                 m.bias.data.fill_(0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def forward(self, images):
#         """
#         images: [B, 3, 224, 224]
#         返回: [B, embed_dim]
#         """
#         with torch.no_grad():
#             # Swin-Transformer前向传播
#             features = self.swin_model(images)  # [B, 1024]
#
#         # 投影到目标维度
#         return self.projection(features)


class SwinGlobalPath(nn.Module):
    """Swin-Transformer全局路径（使用timm库）"""
    def __init__(self, embed_dim=512):
        super().__init__()
        # 创建模型，使用 Swin 用于 CLIP 的权重 (例如 laion400m 数据集)
        # self.swin_model = timm.create_model(
        #     'swin_large_patch4_window7_224',
        #     pretrained=True,  # 使用ImageNet预训练
        #     num_classes=0,  # 移除分类头
        #     global_pool='avg'  # 全局平均池化
        # )

        self.swin_model = timm.create_model(
            'swin_large_patch4_window7_224',
            pretrained=True,  # 使用ImageNet预训练
            pretrained_cfg_overlay=dict(file='swin/swin_large_patch4_window7_224.bin'),
            num_classes=0,  # 移除分类头
            global_pool='avg'  # 全局平均池化
        )

        # self.swin_model = SwinModel.from_pretrained("./swin_weights")

        # 冻结Swin主干（可选）
        for param in self.swin_model.parameters():
            param.requires_grad = False

        # 获取特征维度
        self.feature_dim = self.swin_model.num_features  # 通常是1024

        # 投影到目标维度
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        # self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, images):
        """
        images: [B, 3, 224, 224]
        返回: [B, embed_dim]
        """
        with torch.no_grad():
            # Swin前向传播 - 获取pooler_output作为全局特征
            outputs = self.swin_model(images)
            # 使用pooler_output作为全局图像特征 [B, feature_dim]

        # 投影到目标维度
        projected_features = self.projection(outputs)  # [B, embed_dim]

        return projected_features


class SwinLocalPath(nn.Module):
    """Swin-Transformer局部路径（全冻结）"""
    def __init__(self, embed_dim=512, num_crops=32):
        super().__init__()
        self.num_crops = num_crops

        # self.swin_model = timm.create_model(
        #     'swin_large_patch4_window7_224',
        #     pretrained=True,  # 使用ImageNet预训练
        #     num_classes=0,  # 移除分类头
        #     global_pool='avg'  # 全局平均池化
        # )

        self.swin_model = timm.create_model(
            'swin_large_patch4_window7_224',
            pretrained=True,  # 使用ImageNet预训练
            pretrained_cfg_overlay=dict(file='swin/swin_large_patch4_window7_224.bin'),
            num_classes=0,  # 移除分类头
            global_pool='avg'  # 全局平均池化
        )

        # 在您的 SwinGlobalPath 中使用 model.visual 作为提取器
        # self.swin_model = model.visual

        # 加载预训练的Swin-Transformer模型
        # self.swin_model = SwinModel.from_pretrained("./swin_weights")

        # 冻结所有参数
        for param in self.swin_model.parameters():
            param.requires_grad = False

        # 获取特征维度
        self.feature_dim = self.swin_model.num_features  # 通常是1024

        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        # self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, crops):
        """
        crops: [B, num_crops, 3, 224, 224]
        返回: [B, num_crops, embed_dim]
        """
        batch_size = crops.size(0)
        all_crop_features = []

        for i in range(self.num_crops):
            crop = crops[:, i]  # [B, 3, 224, 224]

            with torch.no_grad():
                # Swin-Transformer前向传播
                crop_features = self.swin_model(crop)  # [B, 1024]

            # 投影到目标维度
            projected_features = self.projection(crop_features)  # [B, embed_dim]
            all_crop_features.append(projected_features)

        return torch.stack(all_crop_features, dim=1)  # [B, num_crops, embed_dim]

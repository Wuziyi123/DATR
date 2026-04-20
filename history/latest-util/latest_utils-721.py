import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.fft
import pywt
import numpy as np
from transformers import BertModel, BertConfig, BertTokenizer, AutoModel
from timm.models.vision_transformer import Block
from tqdm import tqdm
import random
import os
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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

    def forward(self, local_img, local_text, cap_lens, global_img=None, global_text=None):
        """
        改进的相似度计算前向传播：
        0. 根据attention_mask准确计算每个文本的有效长度
        1. 添加L2归一化确保特征稳定性
        2. 优化SCAN注意力计算流程
        3. 修正维度变换逻辑
        """

        n_image = local_img.size(0)
        n_caption = local_text.size(0)
        # B, num_texts, _ = global_text.shape
        text_lengths = cap_lens

        # get enhanced global images by self-attention
        img_ave = torch.mean(local_img, 1)
        l_emb = self.dropout(self.tanh(self.emb_local_bn(self.emb_local_li(local_img))))
        g_emb = self.dropout(self.tanh(self.emb_global_bn(self.emb_global_li(img_ave))))
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights_raw = self.emb_common(common)
        weights = self.softmax(weights_raw.squeeze(2)).unsqueeze(2)
        # compute final image, shape: (batch_size, 1024)
        new_global = (weights * local_img).sum(dim=1)
        img_glo, norm_glo = l2norm_glo(new_global, dim=-1)

        # === 相似度计算 ===
        sim_t_list = []
        sim_v_list = []
        global_sim_list = []
        for i in range(len(text_lengths)):
            n_words = text_lengths[i]  # [B]
            text_feat = local_text[i, :n_words, :].unsqueeze(0)
            text_i_expand = text_feat.repeat(n_image, 1, 1)

            # 双向SCAN注意力
            text_context = SCAN_attention(text_i_expand, local_img, smooth=9.0)  # [B, M, D]
            sim_loc_t = torch.pow(torch.sub(text_context, text_i_expand), 2)
            sim_loc_t = l2norm(self.sim_tranloc_wt(sim_loc_t), dim=-1)
            sim_t_list.append(sim_loc_t)

            text_ave_feat = torch.mean(text_feat, 1)
            text_glo_feat = self.t_global_w(text_feat, text_ave_feat)
            img_context = SCAN_attention(local_img, text_i_expand, smooth=9.0)  # [B, num_crops, D]

            # 计算相似度矩阵
            sim_loc_v = torch.pow(torch.sub(img_context, local_img), 2)
            sim_loc_v = l2norm(self.sim_tranloc_wv(sim_loc_v), dim=-1)

            # === 全局相似度计算 ===
            # cap_glo4par_i = cap_glo_i.repeat(36, 1).unsqueeze(0)
            sim_glo = torch.pow(torch.sub(img_glo, text_glo_feat), 2)
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


# class MultiPositiveContrastiveLoss(nn.Module):
#     def __init__(self, margin=0.2, max_violation=True):
#         super(MultiPositiveContrastiveLoss, self).__init__()
#         self.margin = margin
#         self.max_violation = max_violation
#
#     def compute_similarity(self, outputs, logit_scale):
#         """计算相似度矩阵 (N, 5N)"""
#         global_vis = outputs["global_vis"]  # (N, D)
#         text_feats = outputs["text_feats"]  # (N, D)
#         # 计算相似度矩阵 (N, N)
#         similarity = logit_scale * global_vis @ text_feats.t()
#         return similarity
#
#     def forward(self, outputs, logit_scale):
#         # 计算对角线元素（正样本对的相似度）
#         scores = self.compute_similarity(outputs, logit_scale)
#         device = scores.device
#         diagonal = scores.diag().view(scores.size(0), 1)
#         d1 = diagonal.expand_as(scores)
#         d2 = diagonal.t().expand_as(scores)
#
#         # 文本检索损失：每个图像应与其对应文本最相似
#         cost_s = (self.margin + scores - d1).clamp(min=0)
#         # 图像检索损失：每个文本应与其对应图像最相似
#         cost_im = (self.margin + scores - d2).clamp(min=0)
#
#         # 清除对角线（自身比较）
#         mask = torch.eye(scores.size(0)) > .5
#         I = mask.to(scores.device)
#         cost_s = cost_s.masked_fill_(I, 0)
#         cost_im = cost_im.masked_fill_(I, 0)
#
#         # 选择最大违反项或求和
#         if self.max_violation:
#             cost_s = cost_s.max(1)[0]
#             cost_im = cost_im.max(0)[0]
#
#         return cost_s.sum() + cost_im.sum()


class MultiPositiveContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, margin=0.2, max_violation=True):
        super(MultiPositiveContrastiveLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

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
        return cost_s.sum() + cost_im.sum()


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
        with torch.no_grad():
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
        local_feats = []

        for i in range(self.num_crops):
            crop = crops[:, i]

            # 小波增强
            # wave_crop = self.wavelet(crop)

            # 原始CLIP特征
            with torch.no_grad():
                orig_feat = self.clip.encode_image(crop)
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
        # 解冻最后1层
        for layer in self.bert.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        # 语义增强模块（兼容BERT输出维度）
        embed_dim = self.bert.config.hidden_size  # 动态获取预训练模型的隐藏层维度
        # 添加投影层统一维度为512
        self.projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 512)
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
        # 提取全局文本特征 ([CLS] token)
        global_text_feats = outputs.last_hidden_state[:, 0, :]  # (B*num_texts, 768)
        # 提取局部文本特征 (所有 tokens)
        local_text_feats = outputs.last_hidden_state  # (B*num_texts, seq_len, 768)

        # 投影到512维
        global_text_feats = self.projection(global_text_feats)  # (B*num_texts, 512)
        local_text_feats = self.projection(local_text_feats)  # (B*num_texts, seq_len, 512)

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


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Gumbel


class DifferentiableTokenPruning(nn.Module):
    """
    可微分Token削减模块（基于GroupViT思想改进）
    使用可微分的软分配机制代替硬性Top-K选择
    """
    def __init__(self, embed_dim, num_groups=None, keep_ratio=0.7, min_tokens=5,
                 temperature=0.1, use_gumbel_softmax=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.keep_ratio = keep_ratio
        self.min_tokens = min_tokens
        self.temperature = temperature
        self.use_gumbel_softmax = use_gumbel_softmax

        # 可学习的组中心参数
        if num_groups is None:
            self.num_groups = None  # 动态计算
        else:
            self.num_groups = num_groups
            self.group_centers = nn.Parameter(torch.Tensor(num_groups, embed_dim))
            nn.init.normal_(self.group_centers, mean=0.0, std=0.02)

        # 注意力机制用于计算分配权重
        self.attn_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

        self.init_weights()

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, feats, guidance_feats=None):
        """
        前向传播
        feats: [B, N, D] 待削减的特征
        guidance_feats: [B, M, D] 引导特征（可选）
        返回: 削减后的特征 [B, G, D], 分配矩阵 [B, N, G]
        """
        B, N, D = feats.shape

        # 动态计算目标组数
        if self.num_groups is None:
            G = max(self.min_tokens, int(N * self.keep_ratio))
        else:
            G = self.num_groups

        # 计算分配权重
        if guidance_feats is not None:
            # 使用引导特征计算跨模态注意力
            # [B, N, D] @ [B, D, M] -> [B, N, M]
            cross_attn = torch.bmm(feats, guidance_feats.transpose(1, 2))
            cross_attn = F.softmax(cross_attn / self.temperature, dim=-1)

            # 聚合引导特征 [B, N, M] @ [B, M, D] -> [B, N, D]
            guided_feats = torch.bmm(cross_attn, guidance_feats)

            # 融合原始特征和引导特征
            fused_feats = feats + guided_feats
            attn_scores = self.attn_proj(fused_feats)  # [B, N, 1]
        else:
            # 仅使用自身特征
            attn_scores = self.attn_proj(feats)  # [B, N, 1]

        # 确保分配权重的可微分性
        if self.use_gumbel_softmax and self.training:
            # Gumbel-Softmax重参数化技巧
            attn_scores = attn_scores.squeeze(-1)  # [B, N]

            # 重复G次以匹配目标组数
            attn_scores = attn_scores.unsqueeze(-1).repeat(1, 1, G)  # [B, N, G]

            # 添加Gumbel噪声
            gumbel_dist = Gumbel(0, 1)
            gumbel_noise = gumbel_dist.sample(attn_scores.shape).to(attn_scores.device)
            noisy_scores = (attn_scores + gumbel_noise) / self.temperature

            # Softmax得到分配矩阵
            assignment = F.softmax(noisy_scores, dim=1)  # [B, N, G]
        else:
            # 标准Softmax
            attn_scores = attn_scores.squeeze(-1)  # [B, N]

            # 重复G次以匹配目标组数
            attn_scores = attn_scores.unsqueeze(-1).repeat(1, 1, G)  # [B, N, G]

            # Softmax得到分配矩阵
            assignment = F.softmax(attn_scores / self.temperature, dim=1)  # [B, N, G]

        # 标准化分配矩阵（确保每个组的总权重为1）
        norm_factor = assignment.sum(dim=1, keepdim=True) + 1e-8
        assignment = assignment / norm_factor

        # 使用分配矩阵聚合特征
        pruned_feats = torch.bmm(assignment.transpose(1, 2), feats)  # [B, G, N] @ [B, N, D] -> [B, G, D]

        # 如果提供了组中心，则与聚合特征融合
        if hasattr(self, 'group_centers'):
            # 扩展组中心以匹配批次大小 [G, D] -> [B, G, D]
            group_centers = self.group_centers.unsqueeze(0).repeat(B, 1, 1)
            # 融合组中心和聚合特征
            pruned_feats = pruned_feats + group_centers

        return pruned_feats, assignment


class GroupingBlock(nn.Module):
    """
    GroupViT风格的分组块：多层可微分Token削减
    """

    def __init__(self, embed_dim, num_stages=2, reduction_ratios=[0.75, 0.5]):
        super().__init__()
        self.num_stages = num_stages
        self.grouping_blocks = nn.ModuleList()

        for i in range(num_stages):
            ratio = reduction_ratios[i] if i < len(reduction_ratios) else reduction_ratios[-1]
            self.grouping_blocks.append(
                DifferentiableTokenPruning(embed_dim, keep_ratio=ratio)
            )

        # 可学习的组Token（用于初始分组）
        self.group_tokens = nn.Parameter(torch.Tensor(64, embed_dim))
        nn.init.normal_(self.group_tokens, mean=0.0, std=0.02)

    def forward(self, feats, guidance_feats=None):
        """
        多阶段分组
        feats: [B, N, D] 输入特征
        返回: 分组后的特征 [B, G, D], 所有分配矩阵
        """
        all_assignments = []
        current_feats = feats

        for i, grouping_block in enumerate(self.grouping_blocks):
            # 第一阶段使用可学习的组Token作为引导
            if i == 0 and guidance_feats is None:
                # 扩展组Token以匹配批次大小 [G, D] -> [B, G, D]
                batch_size = feats.size(0)
                group_tokens = self.group_tokens.unsqueeze(0).repeat(batch_size, 1, 1)
                current_feats, assignment = grouping_block(current_feats, group_tokens)
            else:
                current_feats, assignment = grouping_block(current_feats, guidance_feats)

            all_assignments.append(assignment)

        return current_feats, all_assignments

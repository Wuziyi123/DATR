import copy
import os
import random
import time
# 在文档4的开头添加必要的导入
from torch.amp import GradScaler, autocast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.backends import cudnn
from torch.distributions import Gumbel
import torch.fft
import pywt
import numpy as np
from transformers import BertModel, BertConfig, BertTokenizer
from timm.models.vision_transformer import Block
from torch.utils.data import DataLoader
import torch.optim as optim

from data import deserialize_vocab, get_loaders, EncoderImage, EncoderText
from latest_utils import init_weights
from latest_utils import VisualExpert, SemanticEmbeddingLearner, VarianceControlModule, EncoderSimilarity
from latest_utils import (TextEncoder, ContextEnhancer, LocalWaveletPath, LocalPath,
GlobalFourierPath, MultiPositiveContrastiveLoss, SimilarityComputer, WaveViTBlock, set_seed)
from evaluate import (validate, )
from latest_datasetloader import Flickr30kDataset, load_flickr_annotations
from tqdm import tqdm
import clip
from latest_utils import (TextGuidedImagePurification, ImageGuidedTextPurification, AdaptiveTokenPruning,
                          DistributionCalibration, GradualPurificationScheduler,
                          AdvancedDistributionCalibration, CrossModalDistributionAlignment)


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


class AdvancedCrossModalRetriever(nn.Module):
    """高级图文检索网络：集成语义嵌入学习和噪声鲁棒机制"""

    def __init__(self, clip_model, trans_model=None, embed_dim=512, sim_dim=256, num_crops=5, num_semantic_tokens=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_crops = num_crops
        self.sim_dim = sim_dim
        self.mode = "train"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.frozen_trans = copy.deepcopy(clip_model).visual.transformer
        # self.frozen_trans = trans_model.visual

        # === 全局特征路径：傅里叶增强 ===
        self.global_path = GlobalFourierPath(clip_model, embed_dim)
        # === 局部特征路径：小波增强 ===
        # 局部分支 (小波增强)
        # self.local_path = LocalPath(768, 8, self.frozen_trans)
        self.local_path = LocalWaveletPath(trans_model, embed_dim, num_crops)
        # === 文本编码器 ===
        self.text_encoder = TextEncoder(embed_dim)

        # 添加相似度计算器
        self.similarity_computer = SimilarityComputer(embed_dim, mode="train")
        self.sim_enc = EncoderSimilarity(embed_dim, sim_dim)

        # 添加新模块
        self.text_guided_image_purify = TextGuidedImagePurification(embed_dim)
        self.image_guided_text_purify = ImageGuidedTextPurification(embed_dim)
        # 添加净化调度器
        self.purification_scheduler = GradualPurificationScheduler(total_epochs=60)
        # 添加净化强度控制参数
        self.purification_strength = nn.Parameter(torch.tensor(0.1))

        # self.image_token_pruning = AdaptiveTokenPruning(embed_dim, keep_ratio=0.7)
        # self.text_token_pruning = AdaptiveTokenPruning(embed_dim, keep_ratio=0.8)
        # 替换简单的DistributionCalibration为高级版本
        self.distribution_calibration = AdvancedDistributionCalibration(
            embed_dim, mode='both', use_mi=True
        )
        # 添加分布对齐模块
        self.distribution_alignment = CrossModalDistributionAlignment(embed_dim)

        params = list(self.global_path.parameters())
        params += list(self.local_path.parameters())
        params += list(self.text_encoder.parameters())
        params += list(self.similarity_computer.parameters())
        params += list(self.sim_enc.parameters())

        params += list(self.text_guided_image_purify.parameters())
        params += list(self.image_guided_text_purify.parameters())
        params += list(self.distribution_calibration.parameters())
        params += list(self.distribution_alignment.parameters())

        # elf.text_token_pruning.parameters())
        # params += list(self.distribution_calibration.parameters())

        self.params = params

        self.optimizer = torch.optim.AdamW(self.params, lr=1e-5)

        # self.image_global_proj = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.BatchNorm1d(embed_dim),
        #     nn.ReLU(),
        # )
        # self.image_local_proj = nn.Sequential(
        #     nn.Linear(512, embed_dim),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        # )

        # === 特征净化模块 ===
        # self.feature_purifier = CrossModalPurifier(embed_dim)

        # === 多匹配关系建模 ===
        # self.multi_match = MultiMatchModule(embed_dim)

        # === 损失模块 ===
        # self.loss_module = MultiLevelLoss(embed_dim)
        self.criterion = MultiPositiveContrastiveLoss()
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

    def set_epoch(self, epoch):
        """设置当前epoch，用于调整净化强度"""
        self.purification_scheduler.current_epoch = epoch
        # 更新净化模块的门控值
        gate_value = self.purification_scheduler.get_gate_value()
        self.text_guided_image_purify.gate.data.fill_(gate_value)
        self.image_guided_text_purify.gate.data.fill_(gate_value)

    def forward(self, images, texts, attention_mask=None):
        # ===== 特征提取 =====
        # 全局特征 (傅里叶增强)
        global_feat = self.global_path(images[:, 0])
        # 局部特征 (小波增强)
        local_imgs = images[:, 1:]  # (B, N, 3, H, W)
        local_feats = self.local_path(local_imgs)

        # 提取文本特征
        global_text, local_text = self.text_encoder(texts, attention_mask)
        global_text = global_text.squeeze(1)  # (B, 512)
        local_text = local_text.squeeze(1)  # (B, 77, 512)
        del images, texts
        torch.cuda.empty_cache()

        # 计算相似度
        sim_matrix = None
        if self.mode == "train":
            # 预分布校准
            # 全局特征分布校准
            global_feat_cal, global_text_cal, cal_loss1 = self.distribution_calibration(global_feat, global_text)

            # 局部特征分布校准
            # B, N, D = local_feats.shape
            # local_feats_flat = local_feats.view(B * N, D)
            # local_text_flat = local_text.view(B * local_text.size(1), D)
            #
            # local_feats_cal, local_text_cal, cal_loss2 = self.distribution_calibration(local_feats_flat, local_text_flat)
            #
            # local_feats_cal = local_feats_cal.view(B, N, D)
            # local_text_cal = local_text_cal.view(B, local_text.size(1), D)
            #
            # # 分布对齐
            # local_feats_aligned, local_text_aligned, align_loss = self.distribution_alignment(
            #     local_feats_cal, local_text_cal
            # )
            # 组合所有分布损失
            # distribution_loss = cal_loss1 + cal_loss2 + align_loss

            # 使用校准后的特征
            distribution_loss = cal_loss1
            global_feat = global_feat_cal
            global_text = global_text_cal
            # local_feats = local_feats_aligned
            # local_text = local_text_aligned

            # 文本引导图像净化
            # local_feats = self.text_guided_image_purify(local_feats, local_text)
            # 图像引导文本净化
            # local_text = self.image_guided_text_purify(local_text, local_feats)
            # Token自适应削减
            # local_feats, _ = self.image_token_pruning(local_feats, local_text)
            # local_text, _ = self.text_token_pruning(local_text, local_feats)

            global_feat = F.normalize(global_feat, p=2, dim=-1)
            global_text = F.normalize(global_text, p=2, dim=-1)
            local_text = F.normalize(local_text, p=2, dim=-1)
            local_feats = F.normalize(local_feats, p=2, dim=-1)

            sim_t_list, sim_v_list = self.similarity_computer(
                attention_mask.squeeze(1),
                global_feat,
                global_text,  # 添加num_texts维度
                local_feats,
                local_text  # 添加num_texts维度
            )
            # === 最终相似度计算 ===  应用自推理模块计算最终相似度
            sim_matrix = self.sim_enc(sim_t_list, sim_v_list)
            del sim_t_list, sim_v_list
            torch.cuda.empty_cache()
        else:
            global_feat = F.normalize(global_feat, p=2, dim=-1)
            global_text = F.normalize(global_text, p=2, dim=-1)
            local_text = F.normalize(local_text, p=2, dim=-1)
            local_feats = F.normalize(local_feats, p=2, dim=-1)

        return {
            "global_vis": global_feat,
            "local_vis": local_feats,
            "text_feats": global_text,
            "local_text": local_text,
            "sim_matrix": sim_matrix,
            "distribution_loss": distribution_loss if self.mode == "train" else 0.0
        }

    def compute_loss(self, outputs):
        """计算多正样本对比损失"""
        contrastive_loss = self.criterion(outputs)
        if self.mode == "train":
            # 组合对比损失和分布对齐损失
            total_loss = contrastive_loss + 0.1 * outputs["distribution_loss"]
            return total_loss
        else:
            return contrastive_loss

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


# ===== 训练主函数 =====
def adjust_learning_rate(optimizer, epoch):
    """
    Sets the learning rate to the initial LR
    decayed by 10 after opt.lr_update epoch
    """
    initial_lr = 1e-5
    if epoch < 40:
        lr = initial_lr
    # elif epoch < 40:
    #     lr = 5e-5
    else:
        lr = initial_lr / 10  # 1/5 * 1/10 = 1/50
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_retriever():
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # set_seed(0)

    dpath = os.path.join("./DATA/", "f30k_precomp")
    # vocab = deserialize_vocab(os.path.join('./DATA/vocab/', '%s_vocab.json' % 'f30k_precomp'))
    # vocab_size = len(vocab)
    vocab = None

    # 加载CLIP模型
    clip_model, _ = clip.load("ViT-B/16", device=device)
    for param in clip_model.parameters():
        param.requires_grad = False
    for name, param in clip_model.named_parameters():
        if "blocks.11" in name or "blocks.10" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    trans_model, _ = clip.load("ViT-B/16", device=device)
    for param in trans_model.parameters():
        param.requires_grad = False
    for name, param in trans_model.named_parameters():
        if "blocks.11" in name or "blocks.10" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)

    # 初始化数据集
    annotation_path = "flickr30k/results_20130124.token"
    image_dir = "flickr30k/flickr30k-images"
    rpn_file = "flickr30k/flickr30k_rpn_proposals-U.json"
    annotation_df = load_flickr_annotations(annotation_path)

    # 新划分：1000测试集 + 1000评估集 + 其余训练集
    image_ids = list(set(annotation_df['image_id']))
    random.shuffle(image_ids)
    test_image_ids = image_ids[:1000]  # 前1000张测试集
    eval_image_ids = image_ids[1000:2000]  # 后续1000张评估集
    train_image_ids = image_ids[2000:]  # 其余为训练集
    # 创建训练集和验证集数据框
    train_annotations = annotation_df[annotation_df['image_id'].isin(train_image_ids)]
    eval_annotations = annotation_df[annotation_df['image_id'].isin(eval_image_ids)]
    test_annotations = annotation_df[annotation_df['image_id'].isin(test_image_ids)]

    train_dataset = Flickr30kDataset(image_dir, train_annotations, dpath, vocab, 'train', mode='train', rpn_proposals_file=rpn_file)
    eval_dataset = Flickr30kDataset(image_dir, eval_annotations, dpath, vocab,  'dev', mode='eval')
    # test_dataset = Flickr30kDataset(image_dir, test_annotations, dpath, vocab, 'dev', mode='test',)
    # 创建数据加载器
    dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    model = AdvancedCrossModalRetriever(clip_model, trans_model, num_crops=32).to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f}M")

    # 优化器配置
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=1e-4,
    #     weight_decay=1e-2,
    # )

    # 训练循环
    num_epochs = 60
    best_epoch = 0
    val_rsum = 0
    best_test_rsum = 0
    best_rsum = 0
    # 记录所有epoch的测试结果
    all_test_results = []
    accumulation_steps = 4  # 每4个批次更新一次梯度
    total_batches = len(dataloader)

    # 第一阶段：余弦退火 (epoch 0-39)
    # from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR, SequentialLR
    # T_max=40 表示余弦周期为40个epoch，eta_min设置为初始学习率的1/20
    # scheduler_cos = CosineAnnealingLR(optimizer, T_max=40, eta_min=2e-5 / 20)  # eta_min=1e-6

    # 第二阶段：恒定学习率 (epoch 40-59)
    # factor=1.0 表示学习率保持为当前值（即余弦退火结束时的值）不变
    # scheduler_const = ConstantLR(optimizer, factor=1.0, total_iters=20)

    # 使用SequentialLR组合两个调度器
    # [40, 20] 表示第一个调度器运行40个epoch，第二个接着运行20个epoch
    # scheduler = SequentialLR(optimizer, schedulers=[scheduler_cos, scheduler_const], milestones=[40])

    # 添加混合精度训练所需的scaler
    scaler = GradScaler(enabled=(device == "cuda"))  # 仅在CUDA设备上启用

    for epoch in range(num_epochs):
        model.train()
        model.mode = "train"

        model.set_epoch(epoch)  # 设置当前epoch，用于调整净化强度

        total_loss = 0
        adjust_learning_rate(model.optimizer, epoch)
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        epoch_start = time.time()

        # 在每个epoch开始时清零梯度
        model.optimizer.zero_grad()

        for batch_idx, batch in progress_bar:
            # accumulation_count += 1
            images = batch['images'].to(device, non_blocking=True)
            batch_size = images.size(0)
            # 获取多文本数据 [B, 5, 77]
            texts = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)

            # 使用混合精度训练
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # 前向传播
                outputs = model(images, texts, attention_mask)
                loss = model.compute_loss(outputs)
                loss = loss / accumulation_steps  # 损失值按累积步数缩放

            # 反向传播（梯度累积）
            scaler.scale(loss).backward()
            total_loss += loss.item() * accumulation_steps  # 为了日志显示，恢复近似原始损失值
            progress_bar.set_postfix({"loss": loss.item()})

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_batches:
                # 梯度裁剪
                scaler.unscale_(model.optimizer)
                clip_grad_norm_(model.params, 2.0)
                scaler.step(model.optimizer)
                scaler.update()
                model.optimizer.zero_grad()
                del outputs
                torch.cuda.empty_cache()
            else:
                pass

        train_time = time.time() - epoch_start
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
        # scheduler.step()  # 关键：在这里调用调度器更新学习率
        current_lr = model.optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate after Epoch {epoch + 1}: {current_lr:.8f}")

        # ===== 验证阶段 =====
        val_start = time.time()
        model.mode = "eval"
        with torch.no_grad():
            (r1_i2t, r5_i2t, r10_i2t), (r1_t2i, r5_t2i, r10_t2i), val_rsum = validate(model, eval_loader, device)
        print(f"\nEpoch {epoch + 1} Validation Results:")
        print(f"Image to Text: R@1={r1_i2t:.1f}, R@5={r5_i2t:.1f}, R@10={r10_i2t:.1f}")
        print(f"Text to Image: R@1={r1_t2i:.1f}, R@5={r5_t2i:.1f}, R@10={r10_t2i:.1f}")
        print(f"VAL_RSUM: {val_rsum:.1f}")
        val_time = time.time() - val_start

        # ===== 测试阶段 =====
        # test_start = time.time()
        # with torch.no_grad():
        #     test_i2t, test_t2i, test_rsum = validate(model, test_loader, device)
        # test_time = time.time() - test_start

        # 记录测试结果
        # test_results = {
        #     "epoch": epoch + 1,
        #     "r1_i2t": test_i2t[0],
        #     "r5_i2t": test_i2t[1],
        #     "r10_i2t": test_i2t[2],
        #     "r1_t2i": test_t2i[0],
        #     "r5_t2i": test_t2i[1],
        #     "r10_t2i": test_t2i[2],
        #     "rsum": test_rsum
        # }
        # all_test_results.append(test_results)

        # ===== 打印epoch总结 =====
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{num_epochs} Summary:")
        print(f"{'-' * 60}")
        print(f"Training Loss: {avg_loss:.4f} | Time: {train_time:.2f}s")
        print(f"Validation RSUM: {val_rsum:.1f} | Time: {val_time:.2f}s")
        # print(f"Test RSUM: {test_rsum:.1f} | Time: {test_time:.2f}s")
        # print(f"{'-' * 60}")
        # print("Test Results:")
        # print(f"Image to Text: R@1 = {test_i2t[0]:.1f}, R@5 = {test_i2t[1]:.1f}, R@10 = {test_i2t[2]:.1f}")
        # print(f"Text to Image: R@1 = {test_t2i[0]:.1f}, R@5 = {test_t2i[1]:.1f}, R@10 = {test_t2i[2]:.1f}")
        # print(f"{'=' * 60}\n")

        # 保存最佳模型（基于验证集RSUM）
        if val_rsum > best_rsum:
            best_rsum = val_rsum
            best_epoch = epoch + 1
            # best_test_rsum = test_rsum
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'best_rsum': best_rsum,
            }, "retriever_epoch{epoch + 1}.pth")
            print(
                f" New best model saved at epoch {epoch + 1} with val RSUM={val_rsum:.1f}")

    print("Training completed!")


if __name__ == "__main__":
    train_retriever()

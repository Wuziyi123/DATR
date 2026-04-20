import copy
import os
import random
import time
import math
# 在文档4的开头添加必要的导入
from torch.amp import GradScaler, autocast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.backends import cudnn
from torch.distributions import Gumbel
import torch.fft
# import pywt
import numpy as np
from timm.models.vision_transformer import Block
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
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
                          AdvancedDistributionCalibration, GradualPurificationScheduler,
                          SwinGlobalPath, SwinLocalPath)
from latest_utils import IntraModalMetricConsistencyLoss  # 新增导入

from GroupViT import GroupingBlock
# 导入自定义模块
# from model_arch import EnhancedTextEncoder
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import cv2


def load_trained_model(model_path, device, clip_model_path="ViT-L/14@336px"):
    """加载训练好的模型"""
    # 加载CLIP模型
    clip_model, _ = clip.load(clip_model_path, device=device)
    for param in clip_model.parameters():
        param.requires_grad = False
    trans_model, _ = clip.load("ViT-L/14@336px", device=device)
    for param in trans_model.parameters():
        param.requires_grad = False

    # 初始化模型
    model = AdvancedCrossModalRetriever(clip_model, trans_model, num_crops=32).to(device)

    # 加载训练权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.mode = "eval"
    print(f"Model loaded from {model_path}")

    return model


class CustomMultiheadAttention(nn.Module):
    """自定义MultiheadAttention层，支持保存注意力权重和梯度"""
    def __init__(self, embed_dim, num_heads, dropout=0.0, batch_first=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

        # 用于保存中间结果
        self.attention_weights = None  # 保存注意力权重 [batch_size, num_heads, target_len, source_len]
        self.attention_gradients = None  # 保存注意力梯度
        self.grad_hook_handle = None  # 用于管理钩子
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

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True):
        """
        前向传播
        Args:
            query: [batch_size, target_len, embed_dim]
            key: [batch_size, source_len, embed_dim]
            value: [batch_size, source_len, embed_dim]
        Returns:
            attn_output: [batch_size, target_len, embed_dim]
            attn_weights: [batch_size, num_heads, target_len, source_len]
        """
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, target_len, _ = query.shape
        source_len = key.size(1)

        # 线性投影
        q = self.q_proj(query).view(batch_size, target_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用mask（如果有）
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(3), float('-inf')
            )

        # 计算注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        if need_weights:
            # self.attention_weights = attn_weights.detach().clone()
            # self.attention_weights.requires_grad_(True)
            # self.attention_weights.retain_grad()
            self.attention_weights = attn_weights
            # 确保权重保留梯度
            self.attention_weights.requires_grad_(True)
            self.attention_weights.retain_grad()
            # 注册梯度钩子
            self._register_grad_hook()

        # 应用注意力权重到value
        attn_output = torch.matmul(attn_weights, v)

        # 重新排列维度
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, target_len, self.embed_dim
        )
        attn_output = self.out_proj(attn_output)

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        # 返回结果和注意力权重
        if need_weights:
            return attn_output, attn_weights
        else:
            return attn_output, None

    def _register_grad_hook(self,):
        """注册梯度钩子来捕获非叶子节点的梯度"""
        # 移除之前的钩子（如果存在）
        if self.grad_hook_handle is not None:
            self.grad_hook_handle.remove()

        def grad_hook(grad):
            # 保存梯度
            self.attention_gradients = grad
            print(f"[DEBUG] Gradient hook triggered, gradient shape: {grad.shape}")
            # 返回原始梯度，不修改
            return grad

        # 注册钩子
        if self.attention_weights is not None:
            self.grad_hook_handle = self.attention_weights.register_hook(grad_hook)

    def get_attention_weights(self):
        """获取注意力权重"""
        return self.attention_weights

    def get_attention_gradients(self):
        """获取注意力梯度"""
        return self.attention_gradients

    def clear_attention_cache(self):
        """清空注意力缓存"""
        self.attention_weights = None
        self.attention_gradients = None


class AdvancedCrossModalRetriever(nn.Module):
    """高级图文检索网络：集成语义嵌入学习和噪声鲁棒机制"""
    def __init__(self, clip_model, trans_model=None, embed_dim=1024, sim_dim=256, num_crops=5, num_semantic_tokens=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_crops = num_crops
        self.sim_dim = sim_dim
        self.mode = "train"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 添加校准损失权重

        self.align_loss = 0
        # self.frozen_trans = copy.deepcopy(clip_model).visual.transformer
        # self.frozen_trans = trans_model.visual

        # === 全局特征路径 ===
        self.global_path = GlobalFourierPath(clip_model, embed_dim)
        # === 局部特征路径 ===
        # 局部分支
        # self.local_path = LocalPath(768, 8, self.frozen_trans)
        self.local_path = LocalWaveletPath(trans_model, embed_dim, num_crops)

        # swin_pretrained_path = "swin_base_patch4_window7_224_22kto1k.pth"
        # self.global_path = SwinGlobalPath(embed_dim)

        # === 局部特征路径：Swin-Transformer ===
        # self.local_path = SwinLocalPath(embed_dim)

        # === 文本编码器 ===
        self.text_encoder = TextEncoder(embed_dim)
        # self.text_encoder = EnhancedTextEncoder(embed_dim,)

        # self.image_attention_proj = nn.Linear(768, embed_dim)

        # 添加分布对齐模块
        # self.distribution_calibration = AdvancedDistributionCalibration(embed_dim, kl_weight=0.5, mi_weight=0.05)
        self.distribution_calibration = AdvancedDistributionCalibration(
            embed_dim, mode='both', use_mi=True
        )

        # self.image_token_pruning = AdaptiveTokenPruning(embed_dim, keep_ratio=0.7)
        # self.text_token_pruning = AdaptiveTokenPruning(embed_dim, keep_ratio=0.8)
        self.grouping_block = GroupingBlock(embed_dim, num_stages=3, reduction_ratios=[0.9, 0.8, 0.6])
        # self.grouping_block1 = GroupingBlock(embed_dim, num_stages=1, reduction_ratios=[0.9])

        # 添加相似度计算器
        self.similarity_computer = SimilarityComputer(embed_dim, mode="train")
        self.sim_enc = EncoderSimilarity(embed_dim, sim_dim)

        params = list(self.global_path.parameters())
        params += list(self.local_path.parameters())
        params += list(self.text_encoder.parameters())
        # params += list(self.image_attention_proj.parameters())

        params += list(self.distribution_calibration.parameters())

        # params += list(self.text_guided_image_purify.parameters())
        # params += list(self.image_guided_text_purify.parameters())
        params += list(self.grouping_block.parameters())
        # params += list(self.grouping_block1.parameters())
        params += list(self.similarity_computer.parameters())
        params += list(self.sim_enc.parameters())

        # params += list(self.distribution_alignment.parameters())

        # elf.text_token_pruning.parameters())
        # params += list(self.distribution_calibration.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(self.params, lr=1e-4)

        # self.image_global_proj = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.BatchNorm1d(embed_dim),
        #     nn.ReLU(),
        # )

        self.criterion = MultiPositiveContrastiveLoss()
        self.metric_consistency_loss = IntraModalMetricConsistencyLoss(margin=0.2)  # margin α 设为 0.2


        # 注册钩子保存注意力权重和梯度
        # self.attention_weights = None
        # self.attention_gradients = None

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
        # self.image_guided_text_purify.gate.data.fill_(gate_value)

    # def _register_grad_hook(self, attn_weights):
    #     """注册梯度钩子来捕获非叶子节点的梯度"""
    #     # 保存注意力权重
    #     self.attention_weights = attn_weights
    #     attn_weights.retain_grad()  # 关键：确保非叶子节点保留梯度
    #
    #     # 移除之前的钩子（如果存在）
    #     if self.grad_hook_handle is not None:
    #         self.grad_hook_handle.remove()
    #
    #     # 注册新的梯度钩子
    #     def grad_hook(grad):
    #         # 保存梯度
    #         print("Grad hook triggered!")  # 调试信息
    #         self.attention_gradients = grad
    #         # 返回原始梯度，不修改
    #         return grad
    #
    #     # 在非叶子节点上注册钩子
    #     self.grad_hook_handle = attn_weights.register_hook(grad_hook)

    # def _remove_grad_hook(self):
    #     """移除梯度钩子"""
    #     if self.grad_hook_handle is not None:
    #         self.grad_hook_handle.remove()
    #         self.grad_hook_handle = None
    #     self.attention_weights = None
    #     self.attention_gradients = None

    def forward(self, images, texts, attention_mask=None, original_texts=None):
        # if attention_mask is not None:
        #     # 创建key_padding_mask：文本序列的填充位置为True
        #     key_padding_mask = (attention_mask.squeeze(1) == 0).bool()  # [B, seq_len]
        # else:
        #     key_padding_mask = None
        # ===== 特征提取 =====
        # 全局特征
        global_feat = self.global_path(images[:, 0])

        # 局部特征
        local_imgs = images[:, 1:]  # (B, N, 3, H, W)
        local_feats = self.local_path(local_imgs)

        # 提取文本特征
        global_text, local_text = self.text_encoder(texts, attention_mask)
        global_text = global_text.squeeze(1)  # (B, 512)
        local_text = local_text.squeeze(1)  # (B, 77, 512)
        del images, texts
        torch.cuda.empty_cache()

        sim_matrix = None
        if self.mode == "train":
            # 先计算相似度矩阵用于硬负样本选择
            global_feat_cal, global_text_cal, cal_loss1 = self.distribution_calibration(global_feat, global_text)
            global_feat = global_feat_cal
            global_text = global_text_cal
            self.align_loss = cal_loss1

            local_feats, _ = self.grouping_block(local_feats)
            # local_text, _ = self.grouping_block1(local_text)

            global_feat = F.normalize(global_feat, p=2, dim=-1)
            global_text = F.normalize(global_text, p=2, dim=-1)
            local_text = F.normalize(local_text, p=2, dim=-1)
            local_feats = F.normalize(local_feats, p=2, dim=-1)

            sim_t_list, sim_v_list = self.similarity_computer(
                attention_mask.squeeze(1),
                global_feat,
                global_text,
                local_feats,
                local_text
            )
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
            # "itm_loss": itm_loss,
            # "itm_logits": itm_logits
            # "distribution_loss": distribution_loss if self.mode == "train" else 0.0
        }

    def compute_loss(self, outputs, lambda_metric=0.1):
        """计算多正样本对比损失"""
        contrastive_loss = self.criterion(outputs)
        # 新增模态内度量一致性损失
        image_features = outputs["global_vis"]  # 图像特征 [B, D]
        text_features = outputs["text_feats"]  # 文本特征 [B, D]
        metric_loss = self.metric_consistency_loss(image_features, text_features)

        contrastive_loss += lambda_metric * metric_loss

        if self.mode == "train":
            # 组合对比损失和分布对齐损失
            total_loss = contrastive_loss + 0.1 * self.align_loss
            # total_loss = contrastive_loss
            return total_loss
        else:
            return contrastive_loss

    def _register_hooks(self):
        """注册钩子函数来捕获交叉注意力层的权重和梯度"""
        def forward_hook(module, input, output):
            attention_weights = output[1]
            # attention_weights.requires_grad_(True)
            # attention_weights.retain_grad()
            self.attention_weights = attention_weights  # 保存注意力权重 [B, L_img, L_text]

        def backward_hook(module, grad_input, grad_output):
            # 直接从保存的注意力权重中获取梯度
            self.attention_gradients = grad_output[1]

        # 移除旧的钩子，注册新的Grad-CAM专用钩子
        self._remove_hooks()
        self.cross_attention.register_forward_hook(forward_hook)
        self.cross_attention.register_full_backward_hook(backward_hook)

    def _remove_hooks(self):
        """移除所有钩子"""
        if hasattr(self, '_hooks'):
            for hook in self._hooks:
                hook.remove()
        self._hooks = []

    def compute_gradcam(self, image_features, gradcam, attention_mask=None, target_token_idx=None):
        """
        计算Grad-CAM注意力图
        """
        B = 32
        mask = attention_mask.view(B, 1, -1, 1, 1)

        cams = gradcam.activations  # [batch_size, num_heads, target_len, source_len]
        grads = gradcam.gradients  # [batch_size, num_heads, target_len, source_len]

        if cams is None:
            print("Warning: No cams available!")
            return None

        try:
            # 如果梯度仍然为None，使用替代方法
            if grads is None:
                print("Using alternative Grad-CAM method (no gradients available)")
                return self.compute_alternative_gradcam(image_features, target_token_idx)

            # 获取注意力权重和梯度
            # cams = cams[:, :, :, 1:].reshape(B, 8, -1, 14, 14) * mask
            cams = cams[:, :, :, 1:].reshape(B, 8, -1, 14, 14) * mask
            grads_abs = grads[:, :, :, 1:].reshape(B, 8, -1, 14, 14) * mask

            # if grads_abs.max() < 1e-5:  # 如果梯度太小
            #     print("Gradients too small, applying amplification...")
            #     # 归一化到合理范围
            #     grads_normalized = (grads_abs - grads_abs.min()) / (grads_abs.max() - grads_abs.min() + 1e-8)
            #     grads_amplified = grads_normalized * 0.5  # 缩放到[0, 0.5]范围
            #
            #     grads_to_use = grads_amplified
            # else:
            #     grads_to_use = grads_abs

            # 使用加权平均而不是简单相乘
            # 计算每个位置的权重
            weights = F.softmax(grads_abs.view(B, 8, -1), dim=-1)
            weights = weights.view(B, 8, -1, 14, 14)

            # 应用权重到激活图
            gradcam = cams * weights

            # grads = grads[:, :, :, 1:].clamp(0).reshape(B, 8, -1, 14, 14) * mask
            # grads = grads[:, :, :, 1:].abs().reshape(B, 8, -1, 14, 14) * mask

            gradcam = gradcam[0].mean(0).cpu().detach().numpy()

            return gradcam

        except Exception as e:
            print(f"Error computing Grad-CAM: {e}")
            return self.compute_alternative_gradcam(image_features, target_token_idx)

    def visualize(self, original_image, gradcam_map, caption, tokenizer=None, target_token=None, save_path=None):
        """
        可视化Grad-CAM结果
        Args:
            original_image: 原始PIL图像
            gradcam_map: Grad-CAM热力图 [224, 224]
            caption: 对应的字幕文本
            tokenizer: 用于文本标记化的tokenizer
            target_token: 目标token
            save_path: 保存路径
        Returns:
            fig: matplotlib图形对象
        """
        import cv2
        import numpy as np
        from matplotlib import pyplot as plt
        from scipy.ndimage import filters

        def getAttMap(img, attMap, blur=True, overlap=True):
            """生成注意力叠加图"""
            attMap -= attMap.min()
            if attMap.max() > 0:
                attMap /= attMap.max()

            # 调整尺寸匹配原图
            attMap = cv2.resize(attMap, (img.shape[1], img.shape[0]))

            if blur:
                attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
                attMap -= attMap.min()
                if attMap.max() > 0:
                    attMap /= attMap.max()

            cmap = plt.get_cmap('jet')
            attMapV = cmap(attMap)
            attMapV = np.delete(attMapV, 3, 2)

            if overlap:
                attMap = 1 * (1 - attMap ** 0.7).reshape(attMap.shape + (1,)) * img + \
                         (attMap ** 0.7).reshape(attMap.shape + (1,)) * attMapV
            return attMap

        # 转换图像格式
        if isinstance(original_image, Image.Image):
            img_np = np.array(original_image)
        else:
            img_np = original_image

        if len(img_np.shape) == 3 and img_np.shape[-1] == 3:  # RGB转BGR用于OpenCV
            img_np = img_np[:, :, ::-1]
        rgb_image = np.float32(img_np) / 255

        # 处理Grad-CAM图
        if isinstance(gradcam_map, torch.Tensor):
            gradcam_np = gradcam_map.detach().cpu().numpy()
        else:
            gradcam_np = gradcam_map

        # 确保尺寸匹配
        if gradcam_np.shape != rgb_image.shape[:2]:
            gradcam_np = cv2.resize(gradcam_np, (rgb_image.shape[1], rgb_image.shape[0]))

        # 应用Grad-CAM
        atten_map = getAttMap(rgb_image, gradcam_np)

        # 创建可视化
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # 显示原图
        ax1.imshow(cv2.cvtColor(np.uint8(rgb_image * 255), cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image')
        ax1.axis('off')

        # 显示热力图
        im = ax2.imshow(gradcam_np, cmap='jet')
        ax2.set_title('Grad-CAM Heatmap')
        ax2.axis('off')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        # 显示叠加图
        ax3.imshow(atten_map)
        title = f'Grad-CAM Overlay'
        if target_token:
            title += f'\nTarget: {target_token}'
        ax3.set_title(title)
        ax3.axis('off')

        # 添加字幕文本
        fig.suptitle(f'Caption: {caption}', fontsize=12, y=0.95)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to {save_path}")

        return fig

    def visualize_gradcam(self, original_image, gradcam_map, caption, save_path=None):
        """
        可视化Grad-Cam结果，模仿文档4的getAttMap函数
        Args:
            original_image: 原始PIL图像
            gradcam_map: Grad-Cam热力图 [224, 224]
            caption: 对应的字幕文本
            save_path: 保存路径
        """
        import cv2
        import numpy as np
        from matplotlib import pyplot as plt
        from scipy.ndimage import filters
        from skimage import transform as skimage_transform

        def getAttMap(img, attMap, blur=True, overlap=True):
            """修复维度不匹配问题的getAttMap函数"""
            attMap -= attMap.min()
            if attMap.max() > 0:
                attMap /= attMap.max()
            else:
                # 如果全零，创建一个默认的热力图
                attMap = np.ones_like(attMap) * 0.5

            # 调整热力图尺寸匹配原图
            attMap = skimage_transform.resize(attMap, (224, 224), order=3, mode='constant')

            if blur:
                attMap = filters.gaussian_filter(attMap, sigma=1.0)
                attMap -= attMap.min()
                if attMap.max() > 0:
                    attMap /= attMap.max()

            cmap = plt.get_cmap('jet')
            attMapV = cmap(attMap)
            attMapV = np.delete(attMapV, 3, 2)  # 移除alpha通道

            # 确保img是HWC格式 (224,224,3)
            if img.shape[0] == 3:  # 如果是CHW格式 (3,224,224)，转换为HWC
                img = img.transpose(1, 2, 0)

            # 确保attMap是2D，然后扩展为3D用于广播
            attMap_2d = attMap
            if len(attMap.shape) == 3 and attMap.shape[2] == 1:
                attMap_2d = attMap[:, :, 0]

            if overlap:
                # 使用正确的广播维度
                overlay = (1 - attMap_2d ** 0.7)[:, :, np.newaxis] * img + \
                          (attMap_2d ** 0.7)[:, :, np.newaxis] * attMapV
                return overlay
            return attMapV

        # 转换图像格式
        if isinstance(original_image, torch.Tensor):
            img_np = original_image.numpy().transpose(1, 2, 0)  # CHW -> HWC
        else:
            img_np = np.array(original_image)

        # 确保图像是RGB格式
        if img_np.shape[-1] != 3:
            if len(img_np.shape) == 2:  # 灰度图
                img_np = np.stack([img_np] * 3, axis=-1)
            elif img_np.shape[0] == 3:  # CHW格式
                img_np = img_np.transpose(1, 2, 0)  # 转为HWC

        # 归一化到[0,1]
        if img_np.dtype == np.uint8:
            rgb_image = np.float32(img_np) / 255.0
        else:
            rgb_image = np.float32(img_np)

        # 处理Grad-CAM图
        if isinstance(gradcam_map, torch.Tensor):
            gradcam_np = gradcam_map.detach().cpu().numpy()
        else:
            gradcam_np = gradcam_map

        # 选择第一个样本的热力图（如果batch_size>1）
        if len(gradcam_np.shape) > 2:
            cam_to_use = gradcam_np[0]  # 使用第一个样本
        else:
            cam_to_use = gradcam_np

        # 应用Grad-CAM
        try:
            atten_map = getAttMap(rgb_image, cam_to_use)
        except Exception as e:
            print(f"Error in getAttMap: {e}")
            # 创建简单的热力图作为备选
            simple_heatmap = np.ones((224, 224)) * 0.5
            atten_map = getAttMap(rgb_image, simple_heatmap)

        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 显示原图
        ax1.imshow(rgb_image)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # 显示Grad-CAM结果
        ax2.imshow(atten_map)
        ax2.set_title(f'Grad-CAM: {caption}')
        ax2.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to {save_path}")

        return fig


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
    learning_rate = 1e-4
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_retriever():
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    checkpoint = True
    # set_seed(0)

    dpath = os.path.join("./DATA/", "f30k_precomp")
    # vocab = deserialize_vocab(os.path.join('./DATA/vocab/', '%s_vocab.json' % 'f30k_precomp'))
    # vocab_size = len(vocab)
    vocab = None

    # 初始化数据集
    annotation_path = "flickr30k/results_20130124.token"
    image_dir = "flickr30k/flickr30k-images"
    # rpn_file = "flickr30k/flickr30k_rpn_res101_proposals-U.json"
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

    train_dataset = Flickr30kDataset(image_dir, train_annotations, dpath, vocab, 'train', mode='train',
                                     rpn_proposals_file=rpn_file)
    eval_dataset = Flickr30kDataset(image_dir, eval_annotations, dpath, vocab, 'dev', mode='eval',
                                    rpn_proposals_file=rpn_file)
    # test_dataset = Flickr30kDataset(image_dir, test_annotations, dpath, vocab, 'dev', mode='test',)
    # 创建数据加载器
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    if checkpoint:
        # 加载模型
        model_path = "retriever_epoch_best_561-7.pth"  # 替换为您的模型路径
        model = load_trained_model(model_path, device)
    else:
        # 加载CLIP模型
        clip_model, _ = clip.load("ViT-L/14@336px", device=device)
        for param in clip_model.parameters():
            param.requires_grad = False
        # for name, param in clip_model.named_parameters():
        #     if "blocks.11" in name or "blocks.10" in name:
        #         param.requires_grad_(True)
        #     else:
        #         param.requires_grad_(False)

        trans_model, _ = clip.load("ViT-L/14@336px", device=device)
        for param in trans_model.parameters():
            param.requires_grad = False

        # 初始化模型
        model = AdvancedCrossModalRetriever(clip_model, trans_model, num_crops=32).to(device)

    # 打印模型参数数量
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {total_params / 1e6:.2f}M")

    # ===== 添加参数量测算 =====
    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    total_params, trainable_params = count_parameters(model)
    print(f"模型总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f}M")
    print(f"参数量占比: {trainable_params / total_params * 100:.2f}%")

    # 优化器配置
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=1e-4,
    #     weight_decay=1e-2,
    # )

    # 训练循环
    num_epochs = 60
    current_epoch = 15
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

    for epoch in range(current_epoch, num_epochs):
        model.train()
        model.mode = "train"

        # model.set_epoch(epoch)  # 设置当前epoch，用于调整净化强度
        # ===== 添加训练时间测算 =====
        epoch_start_time = time.time()
        torch.cuda.synchronize()  # 确保CUDA操作同步

        total_loss = 0
        adjust_learning_rate(model.optimizer, epoch)
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        # 初始化梯度累积计数器
        accumulation_count = 0
        # 在每个epoch开始时清零梯度
        model.optimizer.zero_grad()

        for batch_idx, batch in progress_bar:
            accumulation_count += 1
            images = batch['images'].to(device, non_blocking=True)
            batch_size = images.size(0)
            # 获取多文本数据 [B, 5, 77]
            texts = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            original_texts = batch['original_text']  # 获取原始文本

            # 使用混合精度训练
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                # 前向传播
                outputs = model(images, texts, attention_mask)
                loss = model.compute_loss(outputs)

            # 反向传播（梯度累积）
            loss = loss / accumulation_steps  # 损失值按累积步数缩放
            scaler.scale(loss).backward()

            if accumulation_count % accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(model.optimizer)
                clip_grad_norm_(model.params, 2.0)
                scaler.step(model.optimizer)
                scaler.update()
                model.optimizer.zero_grad()
                del outputs
                torch.cuda.empty_cache()

            total_loss += loss.item() * accumulation_steps  # 为了日志显示，恢复近似原始损失值
            progress_bar.set_postfix({"loss": loss.item()})

            # 处理剩余批次（当总批次数不是accumulation_steps的倍数时）
        if accumulation_count % accumulation_steps != 0:
            scaler.unscale_(model.optimizer)
            clip_grad_norm_(model.params, 2.0)
            scaler.step(model.optimizer)
            scaler.update()
            model.optimizer.zero_grad()
            del outputs
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
        # scheduler.step()  # 关键：在这里调用调度器更新学习率

        # ===== 计算训练时间 =====
        torch.cuda.synchronize()
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        train_time = epoch_duration  # 保持原有变量名兼容
        avg_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch + 1} 训练时间: {epoch_duration:.2f}秒")
        print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

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

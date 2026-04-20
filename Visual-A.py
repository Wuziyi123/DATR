import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from scipy.ndimage import filters
from skimage import transform as skimage_transform
import os
import json
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 导入您的模型和相关模块
from latest import AdvancedCrossModalRetriever, Flickr30kDataset, load_flickr_annotations
from latest_datasetloader import transform, transform_random_crop


class GradCAMVisualizer:
    def __init__(self, model, device, tokenizer):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.attention_maps = None
        self.gradients = None

        # 注册钩子来获取注意力图和梯度
        self._register_hooks()

    def _register_hooks(self):
        """注册钩子来获取中间层的输出和梯度"""

        def forward_hook(module, input, output):
            self.attention_maps = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # 找到文本编码器中的交叉注意力层
        # 这里假设使用BERT的最后一层交叉注意力
        target_layer = None
        for name, module in self.model.text_encoder.named_modules():
            if 'crossattention' in name.lower() or 'cross_attn' in name.lower():
                target_layer = module
                break

        if target_layer is None:
            # 如果找不到交叉注意力层，使用最后一个Transformer层
            for name, module in self.model.text_encoder.named_modules():
                if isinstance(module, nn.TransformerEncoderLayer):
                    target_layer = module

        if target_layer:
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_backward_hook(backward_hook)
        else:
            print("Warning: Could not find suitable attention layer")

    def preprocess_image(self, image_path, transform_fn):
        """预处理图像"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform_fn(image).unsqueeze(0)
        return image_tensor.to(self.device), image

    def preprocess_text(self, text):
        """预处理文本"""
        text_input = self.tokenizer(
            text,
            padding='max_length',
            max_length=77,
            truncation=True,
            return_tensors='pt'
        )
        return text_input.to(self.device)

    def get_att_map(self, img, attMap, blur=True, overlap=True):
        """生成注意力热力图（从文档1复制）"""
        attMap -= attMap.min()
        if attMap.max() > 0:
            attMap /= attMap.max()
        attMap = skimage_transform.resize(attMap, (img.shape[:2]), order=3, mode='constant')
        if blur:
            attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
            attMap -= attMap.min()
            attMap /= attMap.max()
        cmap = plt.get_cmap('jet')
        attMapV = cmap(attMap)
        attMapV = np.delete(attMapV, 3, 2)
        if overlap:
            attMap = 1 * (1 - attMap ** 0.7).reshape(attMap.shape + (1,)) * img + (attMap ** 0.7).reshape(
                attMap.shape + (1,)) * attMapV
        return attMap

    def compute_gradcam(self, image_tensor, text_input, target_token_idx=None):
        """计算Grad-CAM"""
        # 设置为训练模式以获取梯度
        self.model.train()

        # 前向传播
        outputs = self.model(image_tensor, text_input)

        # 如果未指定目标token，使用[CLS] token
        if target_token_idx is None:
            target_token_idx = 0

        # 计算目标token的梯度
        self.model.zero_grad()
        loss = outputs["sim_matrix"][0, 0]  # 使用相似度矩阵的第一个元素作为目标
        loss.backward()

        if self.attention_maps is None or self.gradients is None:
            print("Warning: No attention maps or gradients captured")
            return None

        # 计算权重
        weights = torch.mean(self.gradients, dim=[1, 2], keepdim=True)

        # 计算加权的注意力图
        cam = torch.sum(weights * self.attention_maps, dim=1)
        cam = F.relu(cam)

        # 调整大小到图像尺寸
        cam = cam.squeeze().cpu().detach().numpy()

        return cam

    def visualize(self, image_path, caption, target_token_idx=None, save_path=None):
        """完整的可视化流程"""
        # 预处理
        image_tensor, orig_image = self.preprocess_image(image_path, transform)
        text_input = self.preprocess_text(caption)

        # 计算Grad-CAM
        cam = self.compute_gradcam(image_tensor, text_input, target_token_idx)

        if cam is None:
            return

        # 准备原始图像
        orig_image_np = np.array(orig_image)
        orig_image_np = orig_image_np.astype(np.float32) / 255

        # 生成热力图
        heatmap = self.get_att_map(orig_image_np, cam)

        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始图像
        axes[0].imshow(orig_image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 注意力热力图
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Attention Heatmap')
        axes[1].axis('off')

        # 叠加结果
        axes[2].imshow(heatmap)
        axes[2].set_title('Overlay Result')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        plt.show()

        return heatmap

    def visualize_multiple_tokens(self, image_path, caption, token_indices=None, save_path=None):
        """可视化多个token的注意力"""
        if token_indices is None:
            # 自动选择重要的token
            text_input = self.preprocess_text(caption)
            token_ids = text_input['input_ids'][0]
            token_indices = []
            for i, token_id in enumerate(token_ids):
                if token_id != 0:  # 排除padding
                    token_indices.append(i)

        # 为每个token生成可视化
        n_tokens = len(token_indices)
        fig, axes = plt.subplots(n_tokens, 3, figsize=(15, 5 * n_tokens))

        if n_tokens == 1:
            axes = [axes]

        image_tensor, orig_image = self.preprocess_image(image_path, transform)
        orig_image_np = np.array(orig_image).astype(np.float32) / 255

        for idx, (ax_row, token_idx) in enumerate(zip(axes, token_indices)):
            # 计算当前token的Grad-CAM
            cam = self.compute_gradcam(image_tensor, self.preprocess_text(caption), token_idx)

            if cam is None:
                continue

            # 获取token文本
            token_text = self.tokenizer.decode([text_input['input_ids'][0][token_idx]])

            # 原始图像
            ax_row[0].imshow(orig_image_np)
            ax_row[0].set_title(f'Original Image\nToken: {token_text}')
            ax_row[0].axis('off')

            # 注意力热力图
            ax_row[1].imshow(cam, cmap='jet')
            ax_row[1].set_title(f'Attention Heatmap\nToken: {token_text}')
            ax_row[1].axis('off')

            # 叠加结果
            heatmap = self.get_att_map(orig_image_np, cam)
            ax_row[2].imshow(heatmap)
            ax_row[2].set_title(f'Overlay Result\nToken: {token_text}')
            ax_row[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved multi-token visualization to {save_path}")

        plt.show()


def load_model_and_tokenizer(model_path, device):
    """加载训练好的模型和tokenizer"""
    # 加载CLIP模型作为基础（根据您的代码）
    import clip
    clip_model, _ = clip.load("ViT-B/16", device=device)
    trans_model, _ = clip.load("ViT-B/16", device=device)

    # 初始化模型
    model = AdvancedCrossModalRetriever(clip_model, trans_model).to(device)

    # 加载训练权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载tokenizer（根据您的代码使用BERT tokenizer）
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('./my_bert')

    return model, tokenizer


def main():
    # 配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "retriever_epoch_best.pth"  # 预训练模型路径
    image_dir = "flickr30k/flickr30k-images"
    annotation_path = "flickr30k/results_20130124.token"

    # 加载模型和tokenizer
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_path, device)

    # 初始化可视化器
    visualizer = GradCAMVisualizer(model, device, tokenizer)

    # 加载数据集信息（用于获取图像和标注）
    annotations = load_flickr_annotations(annotation_path)

    # 选择示例图像和字幕
    sample_image_id = annotations['image_id'].iloc[0]  # 第一个图像
    sample_caption = annotations['caption'].iloc[0]
    image_path = os.path.join(image_dir, sample_image_id)

    print(f"Visualizing image: {sample_image_id}")
    print(f"Caption: {sample_caption}")

    # 执行可视化
    visualizer.visualize(
        image_path=image_path,
        caption=sample_caption,
        save_path="gradcam_visualization.png"
    )

    # 多token可视化
    # visualizer.visualize_multiple_tokens(
    #     image_path=image_path,
    #     caption=sample_caption,
    #     save_path="multi_token_visualization.png"
    # )


if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import re
import json
import pandas as pd
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from transformers import BertModel, BertConfig, BertTokenizer
import torchvision.transforms as transforms
from skimage import transform as skimage_transform
from PIL import Image
import random

# 导入必要的模块
from latest_datasetloader import Flickr30kDataset, load_flickr_annotations
from latest import AdvancedCrossModalRetriever
from data import deserialize_vocab
import clip
from latest import CustomMultiheadAttention


def transform(image):
    """全局图像变换"""
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])(image)


def transform_random_crop(image):
    """局部图像随机裁剪变换"""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 0.9)),
        transforms.RandomHorizontalFlip(),
        lambda x: x.convert("RGB"),  # 替换 _convert_image_to_rgb
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])(image)


class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.gradients = None
        self.activations = None
        self.target_layer = None
        self._register_hooks(target_layer_name)

    def _register_hooks(self, target_layer_name):
        """注册钩子到自定义MultiheadAttention层"""

        def forward_hook(module, input, output):
            # 自定义MultiheadAttention层已经保存了注意力权重
            if hasattr(module, 'attention_weights') and module.attention_weights is not None:
                self.activations = module.attention_weights
                print(f"[DEBUG] Forward hook triggered")
                print(f"[DEBUG] Activation shape: {module.attention_weights.shape}")

        # 找到目标层并注册钩子
        for name, module in self.model.named_modules():
            if name == target_layer_name:
                print(f"[DEBUG] Found target layer: {name}")
                print(f"[DEBUG] Layer type: {type(module)}")
                self.target_layer = module
                # 这里注册额外的钩子用于调试
                if isinstance(module, CustomMultiheadAttention):
                    module.register_forward_hook(forward_hook)
                break

    def get_gradients(self):
        """获取梯度（在反向传播后调用）"""
        if hasattr(self.target_layer, 'attention_gradients'):
            self.gradients = self.target_layer.attention_gradients
            return self.gradients
        return None


def load_trained_model(model_path, device, clip_model_path="ViT-B/16"):
    """加载训练好的模型"""
    # 加载CLIP模型
    clip_model, _ = clip.load(clip_model_path, device=device)
    for param in clip_model.parameters():
        param.requires_grad = False
    trans_model, _ = clip.load("ViT-B/16", device=device)
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

    model.train()
    model.mode = "train"
    print(f"Model loaded from {model_path}")

    return model


def pre_caption(caption, max_words=30):
    """预处理文本，保持与CLIP一致的最大长度"""
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])
    return caption


def tokenize_with_clip_format(text, tokenizer, max_length=77):
    """使用CLIP格式的tokenization，序列长度为77"""
    # 首先使用BERT tokenizer进行基本tokenization
    encoded = tokenizer(text, return_tensors="pt",
                        max_length=max_length,
                        padding='max_length',
                        truncation=True)

    # 获取input_ids和attention_mask
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    return input_ids, attention_mask


def get_valid_tokens_indices(input_ids, tokenizer):
    """获取有效token的索引（过滤特殊token和padding）"""
    valid_indices = []
    valid_tokens = []

    for i, token_id in enumerate(input_ids[0][0]):
        token = tokenizer.decode([token_id])

        # 过滤特殊token：[CLS], [SEP], [PAD]等
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue

        if token.strip() in ['.', '。']:
            continue

        # 过滤标点符号和空白字符
        if token.strip() and len(token.strip()) > 0 and not token.isspace():
            valid_indices.append(i)
            valid_tokens.append(token)

    return valid_indices, valid_tokens


def applyThreshold(attMap, threshold=0.3):
    """应用固定阈值"""
    attMap = np.maximum(attMap - threshold, 0) / (1 - threshold + 1e-8)
    return attMap

def applyAdaptiveThreshold(attMap, adaptive_factor=0.5):
    """应用自适应阈值（基于均值和标准差）"""
    mean_val = attMap.mean()
    std_val = attMap.std()
    threshold = mean_val + adaptive_factor * std_val
    return applyThreshold(attMap, threshold)


def getAttMap(img, attMap, blur=True, overlap=True, threshold=0.3, enhance_contrast=True):
    """getAttMap函数，用于生成注意力叠加图"""
    attMap = applyAdaptiveThreshold(attMap)

    # 创建边缘抑制mask
    h, w = attMap.shape
    edge_mask = np.ones((h, w))
    border_size = max(1, int(0.03 * min(h, w)))  # 至少1像素

    # 边缘区域权重递减
    for i in range(border_size):
        weight = 0.3 + 0.7 * (i / border_size)  # 从0.3线性增加到1.0
        edge_mask[i, :] = np.minimum(edge_mask[i, :], weight)
        edge_mask[-(i + 1), :] = np.minimum(edge_mask[-(i + 1), :], weight)
        edge_mask[:, i] = np.minimum(edge_mask[:, i], weight)
        edge_mask[:, -(i + 1)] = np.minimum(edge_mask[:, -(i + 1)], weight)

    attMap = attMap * edge_mask

    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()

    # if threshold > 0:
    #     # 软阈值 - 逐渐抑制低值区域
    #     attMap = np.maximum(attMap - threshold, 0) / (1 - threshold + 1e-8)

    # 调整热力图尺寸匹配原图
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order=3, mode='constant')

    if blur:
        attMap = gaussian_filter(attMap, 0.02 * max(img.shape[:2]), mode='reflect')
        attMap -= attMap.min()
        if attMap.max() > 0:
            attMap /= attMap.max()

    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)

    if overlap:
        # attMap = np.clip(attMap, 0, 1)  # 显式裁剪到 [0, 1]
        attMap = 1 * (1 - attMap ** 0.7).reshape(attMap.shape + (1,)) * img + \
                 (attMap ** 0.7).reshape(attMap.shape + (1,)) * attMapV
        # attMap = np.clip(attMap, 0, 1)
    return attMap


def visualize_word_attention(image, gradcam_map, valid_tokens, valid_indices,
                             original_image_path, caption, save_path, attention_weights=None):
    """优化后的逐个单词可视化函数"""
    # 加载原始图像
    num_words = len(valid_tokens)
    original_img = cv2.imread(original_image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img_resized = cv2.resize(original_img, (224, 224))
    rgb_image = np.float32(original_img_resized) / 255

    # 根据是否提供注意力权重选择融合方法
    if attention_weights is not None:
        # 使用注意力权重进行加权融合
        combined_heatmap = get_attention_weighted_heatmap(gradcam_map, valid_indices, attention_weights)
    else:
        # 使用几何平均作为备选方案
        combined_heatmap = np.exp(np.mean(np.log(gradcam_map[1:num_words + 1] + 1e-8), axis=0))

    # 生成热图叠加图像
    # combined_heatmap = np.mean(gradcam_map[1:num_words + 1], axis=0)
    heatmap_overlay = getAttMap(rgb_image, combined_heatmap)

    # 单独保存热图叠加图像（不包含原图）
    heatmap_save_path = save_path.replace('.png', '_heatmap_only.png')
    plt.figure(figsize=(10, 10))
    plt.imshow(heatmap_overlay)
    plt.axis('off')  # 隐藏坐标轴
    plt.tight_layout(pad=0)
    plt.savefig(heatmap_save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"单独保存热图叠加图像到: {heatmap_save_path}")

    # 创建可视化图
    # fig, axes = plt.subplots(num_words + 1, 1, figsize=(15, 5 * (num_words + 1)))
    fig, axes = plt.subplots(2, 1, figsize=(15, 5 * 2))

    # 显示原始图像
    axes[0].imshow(rgb_image)
    axes[0].set_title('Original Image')
    axes[0].set_ylabel("Image")
    axes[0].set_yticks([])
    axes[0].set_xticks([])

    # 为每个有效单词显示注意力热图
    for i, (token, idx) in enumerate(zip(valid_tokens, valid_indices)):
        if i >= len(axes) - 1:  # 防止索引越界
            break

        # 使用getAttMap生成叠加图像 1-2 4 6-7
        # gradcam_image = getAttMap(rgb_image, gradcam_map[i + 1])
        gradcam_map = np.mean(gradcam_map[1:num_words+1], axis=0)
        gradcam_image = getAttMap(rgb_image, gradcam_map)

        # 显示结果
        axes[i + 1].imshow(gradcam_image)
        axes[i + 1].set_title(f'Token: "{token}"')
        axes[i + 1].set_ylabel(token)
        axes[i + 1].set_yticks([])
        axes[i + 1].set_xticks([])

        break

    # 设置总标题
    fig.suptitle(f'Caption: {caption}', fontsize=16, y=0.95)
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved word attention visualization to: {save_path}")
    return fig


def get_attention_weighted_heatmap(gradcam_map, valid_indices, attention_weights):
    """
    使用注意力权重进行加权融合
    Args:
        gradcam_map: [target_len, H, W] 每个token的Grad-CAM热力图
        valid_indices: 有效单词的索引列表
        attention_weights: [batch_size, num_heads, target_len, source_len] 注意力权重
    """
    batch_size, num_heads, target_len, source_len = attention_weights.shape
    attn_weights = attention_weights[0]

    # 计算每个token对图像的平均注意力（跨头和图像token维度）
    image_attention = attn_weights[:, :, :]  # [num_heads, target_len, image_tokens_len]

    # 对图像token维度求平均，得到每个文本token对图像的整体注意力
    token_importance = image_attention.mean(dim=(0, 2))  # [target_len]

    # 提取有效单词的注意力权重
    valid_weights = token_importance[valid_indices]

    # 归一化权重
    if valid_weights.sum() > 0:
        normalized_weights = valid_weights / valid_weights.sum()
    else:
        # 如果所有权重为0，使用均匀分布
        normalized_weights = torch.ones_like(valid_weights) / len(valid_weights)

    # 将权重转换为numpy
    normalized_weights = normalized_weights.detach().cpu().numpy()

    # 加权融合热力图
    weighted_heatmap = np.zeros_like(gradcam_map[0])
    for i, weight in enumerate(normalized_weights):
        token_idx = valid_indices[i]
        weighted_heatmap += weight * gradcam_map[token_idx]

    return weighted_heatmap


def create_batch_dataset(image_ids, image_dir, annotation_path, rpn_file, tokenizer, target_image_id, batch_size=32):
    """创建包含多个图像的批次数据集"""
    # 加载注解数据
    annotation_df = load_flickr_annotations(annotation_path)

    # 筛选指定图像ID的注解
    image_annotations = annotation_df[annotation_df['image_id'].isin(image_ids)]

    if len(image_annotations) == 0:
        raise ValueError(f"Image IDs not found in annotations")

    # 创建批次数据集
    class BatchDataset(torch.utils.data.Dataset):
        def __init__(self, image_ids, image_dir, annotations, rpn_file, tokenizer, batch_size, target_image_id):
            self.image_ids = image_ids
            self.image_dir = image_dir
            self.annotations = annotations
            self.tokenizer = tokenizer
            self.rpn_proposals = None
            self.num_crops = 32
            self.rpn_ratio = 0.5
            self.batch_size = batch_size

            # 加载RPN建议框
            if rpn_file and os.path.exists(rpn_file):
                with open(rpn_file, 'r') as f:
                    self.rpn_proposals = json.load(f)

        def __len__(self):
            return 1  # 只返回一个批次

        def __getitem__(self, idx):
            batch_images = []
            batch_input_ids = []
            batch_attention_mask = []
            batch_image_ids = []
            batch_original_texts = []

            # 为每个图像ID创建样本
            for image_id in self.image_ids[:self.batch_size]:  # 确保不超过批次大小
                img_path = os.path.join(self.image_dir, image_id)

                try:
                    # 加载图像
                    image = Image.open(img_path).convert('RGB')
                    w, h = image.size
                    global_img = transform(image)

                    crop_imgs = []

                    # 计算原图总面积
                    original_area = w * h

                    # 筛选面积大于15%原图面积的提案框
                    valid_proposals = []
                    if self.rpn_proposals:
                        # 获取RPN提案
                        proposals = self.rpn_proposals.get(image_id.split('.')[0], [])
                        for box in proposals:
                            x1, y1, x2, y2 = box
                            box_w = x2 - x1
                            box_h = y2 - y1
                            box_area = box_w * box_h
                            # 仅保留面积大于15%原图面积的提案框
                            if box_area >= 0.15 * original_area:
                                valid_proposals.append(box)

                    # 计算RPN建议框和随机裁剪的数量
                    num_rpn = int(self.num_crops * self.rpn_ratio)
                    num_random = self.num_crops - num_rpn

                    # 使用RPN建议框
                    if self.rpn_proposals and len(valid_proposals) > 0 and num_rpn > 0:
                        actual_rpn = min(num_rpn, len(valid_proposals))
                        indices = np.random.choice(len(valid_proposals), actual_rpn, replace=False)
                        for i in indices:
                            x1, y1, x2, y2 = valid_proposals[i]
                            crop_img = image.crop((x1, y1, x2, y2))
                            crop_img = transform(crop_img)
                            crop_imgs.append(crop_img)
                    else:
                        actual_rpn = 0

                    # 使用随机裁剪补足数量
                    for _ in range(num_random + (num_rpn - actual_rpn)):
                        crop_imgs.append(transform_random_crop(image))

                    random.shuffle(crop_imgs)
                    crop_imgs = torch.stack(crop_imgs)
                    images = torch.cat([global_img.unsqueeze(0), crop_imgs], dim=0)
                    batch_images.append(images)

                    # 处理文本
                    image_annots = self.annotations[self.annotations['image_id'] == image_id]
                    captions = image_annots['caption'].tolist()
                    selected_captions = [pre_caption(captions[0])]  # 使用最后一个字幕
                    original_text = pre_caption(captions[0])

                    if image_id == target_image_id:
                        for i in range(len(captions)):
                            print(captions[i])

                    input_ids_list = []
                    attention_mask_list = []
                    for caption in selected_captions:
                        text_input = self.tokenizer(
                            caption,
                            padding='max_length',
                            max_length=77,
                            truncation=True,
                            return_tensors='pt'
                        )
                        input_ids_list.append(text_input['input_ids'])
                        attention_mask_list.append(text_input['attention_mask'])

                    # 堆叠文本
                    input_ids = torch.cat(input_ids_list, dim=0)
                    attention_mask = torch.cat(attention_mask_list, dim=0)

                    batch_input_ids.append(input_ids)
                    batch_attention_mask.append(attention_mask)
                    batch_image_ids.append(image_id)
                    batch_original_texts.append(original_text)

                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue

            # 将批次数据堆叠
            batch_images = torch.stack(batch_images)
            batch_input_ids = torch.stack(batch_input_ids)
            batch_attention_mask = torch.stack(batch_attention_mask)

            return {
                'images': batch_images,
                'input_ids': batch_input_ids,
                'attention_mask': batch_attention_mask,
                'image_ids': batch_image_ids,
                'original_texts': batch_original_texts
            }

    return BatchDataset(image_ids, image_dir, image_annotations, rpn_file, tokenizer, batch_size, target_image_id)


def main():
    # 设备配置
    from latest_utils import set_seed
    # set_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 模型路径
    model_path = "retriever_epoch_best3.pth"  # 替换为您的模型路径

    # 数据路径
    annotation_path = "flickr30k/results_20130124.token"
    image_dir = "flickr30k/flickr30k-images"
    rpn_file = "flickr30k/flickr30k_rpn_proposals-U.json"

    # 选择一批图像ID（包括目标图像和其他31个图像）
    annotation_df = load_flickr_annotations(annotation_path)
    all_image_ids = list(set(annotation_df['image_id']))
    random.shuffle(all_image_ids)

    # 确保目标图像在批次中 7739176066.jpg 3591461782.jpg
    # 已被选中: 440165853.jpg 7238675644.jpg 7739176066.jpg 8404753.jpg 5142968059.jpg
    # 选中: 2833582518.jpg 3039214579.jpg 7099370205.jpg 4060147279.jpg 3242007318.jpg
    # 选中: 3420338549.jpg 263231469.jpg  2078020414.jpg 4442320934.jpg 162759228.jpg
    # 选中: 3440160917.jpg 4393321765.jpg 7645711984.jpg 7955537470.jpg 2934379210.jpg
    # 选中: 2445442929.jpg 2837804631.jpg
    # -142078565.jpg -4661610976.jpg -7955537470.jpg
    target_image_id = '4060147279.jpg'
    if target_image_id not in all_image_ids:
        print(f"Warning: Target image {target_image_id} not found in dataset")
        # 使用其他图像
        batch_image_ids = all_image_ids[:32]
    else:
        # 确保目标图像在批次中，并添加其他31个图像
        batch_image_ids = [target_image_id] + [id for id in all_image_ids if id != target_image_id][:31]

    print(f"Batch image IDs: {batch_image_ids}")

    # 加载模型
    model = load_trained_model(model_path, device)

    # 注意：不再冻结BatchNorm，保持模型在训练模式

    # 创建输出目录
    os.makedirs("gradcam_results", exist_ok=True)

    # 加载tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained('./my_bert')
    except:
        print("Warning: Could not load tokenizer, using simple word splitting")
        tokenizer = None

    # 创建批次数据集
    batch_size = 32
    dataset = create_batch_dataset(batch_image_ids, image_dir, annotation_path,
                                   rpn_file, tokenizer, target_image_id, batch_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)  # 批次大小为1，因为数据集只返回一个批次

    target_layer = 'cross_attention'
    gradcam = GradCAM(model, target_layer)

    try:
        # 获取数据批次
        for batch in dataloader:
            if batch is None:
                continue

            images = batch['images'].squeeze(0).to(device)  # 移除额外的批次维度
            input_ids = batch['input_ids'].squeeze(0).to(device)
            attention_mask = batch['attention_mask'].squeeze(0).to(device)
            image_ids = batch['image_ids']  # 获取图像ID列表
            original_texts = batch['original_texts']  # 获取原始文本列表
            # original_texts = [str(text) for text in original_texts]

            # 修改为：将元组转换为纯字符串列表
            original_texts = [text[0] if isinstance(text, tuple) and len(text) > 0 else str(text) for text in
                              batch['original_texts']]  # 获取原始文本列表

            print(f"Processing batch with {len(image_ids)} images")
            print(f"First image ID: {image_ids[0]}")
            print(f"Image tensor shape: {images.shape}")  # 应该是 [32, 33, 3, 224, 224]

            # 前向传播（整个批次）
            model.zero_grad()

            outputs = model(images, input_ids, attention_mask, original_texts)
            loss = model.compute_loss(outputs)

            print("loss: ", loss)

            # 反向传播获取梯度
            loss.backward(retain_graph=True)

            # 获取梯度信息
            gradients = gradcam.get_gradients()
            print(f"Gradients stats - max: {gradients.max().item() if gradients is not None else 'None'}, "
                  f"min: {gradients.min().item() if gradients is not None else 'None'}, "
                  f"mean: {gradients.mean().item() if gradients is not None else 'None'}")

            # 只对第一个样本（目标图像）进行可视化
            if target_image_id in image_ids:
                target_idx = image_ids.index(target_image_id)
            else:
                target_idx = 0  # 如果目标图像不在批次中，使用第一个图像

            target_image_id_vis = image_ids[target_idx][0]
            target_text = original_texts[target_idx]

            print(f"Visualizing image: {target_image_id_vis}")
            print(f"Caption: {target_text}")

            # 获取有效token的索引（只针对目标样本）
            target_input_ids = input_ids[target_idx:target_idx + 1]  # 保持批次维度
            valid_indices, valid_tokens = get_valid_tokens_indices(input_ids.cpu(), tokenizer)

            print(f"Valid tokens: {valid_tokens}")
            print(f"Valid indices: {valid_indices}")

            # 使用compute_gradcam函数计算Grad-CAM（只针对目标样本）
            if hasattr(model, 'compute_gradcam'):
                # 提取目标样本的图像特征
                target_images = images[target_idx:target_idx + 1]  # 保持批次维度
                global_feat, x1 = model.global_path(target_images[:, 0])
                x1 = model.image_attention_proj(x1)

                # 计算Grad-CAM（只针对目标样本）
                # target_attention_mask = attention_mask[target_idx:target_idx + 1]
                gradcam_map = model.compute_gradcam(
                    x1,
                    gradcam,
                    attention_mask=attention_mask,
                    target_token_idx=None
                )

                if gradcam_map is not None:
                    # 可视化目标样本的单词注意力
                    save_path = f"gradcam_results/batch_sample_{target_image_id_vis.replace('.', '_')}_word_attention.png"

                    target_attention_weights = gradcam.activations[target_idx:target_idx + 1]
                    # 将gradcam_map传递给visualize_word_attention函数
                    fig = visualize_word_attention(
                        target_images, gradcam_map, valid_tokens, valid_indices,
                        os.path.join(image_dir, target_image_id_vis), target_text, save_path,
                        attention_weights=target_attention_weights
                    )

                    print(f"Word attention visualization saved to: {save_path}")

            break  # 只处理一个批次

    except Exception as e:
        print(f"Error processing batch: {e}")
        import traceback
        traceback.print_exc()

    print("Batch Grad-CAM visualization completed!")


if __name__ == "__main__":
    main()

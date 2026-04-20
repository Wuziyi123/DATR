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
from scipy.ndimage import filters
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


def set_bn_training_mode(model, training=True, use_running_stats=True):
    """设置BatchNorm层的行为，保持训练模式但使用运行统计量"""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


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


def getAttMap(img, attMap, blur=True, overlap=True):
    """getAttMap函数，用于生成注意力叠加图"""
    attMap -= attMap.min()
    print(attMap.max())
    if attMap.max() > 0:
        attMap /= attMap.max()
    # 调整热力图尺寸匹配原图
    # attMap = cv2.resize(attMap, (img.shape[1], img.shape[0]))
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order=3, mode='constant')
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02 * max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1 * (1 - attMap ** 0.7).reshape(attMap.shape + (1,)) * img + \
                 (attMap ** 0.7).reshape(attMap.shape + (1,)) * attMapV
    return attMap


def visualize_word_attention(image, gradcam_map, valid_tokens, valid_indices,
                             original_image_path, caption, save_path):
    """优化后的逐个单词可视化函数"""
    # 加载原始图像
    original_img = cv2.imread(original_image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img_resized = cv2.resize(original_img, (224, 224))
    rgb_image = np.float32(original_img_resized) / 255

    # 创建可视化图
    num_words = len(valid_tokens)
    fig, axes = plt.subplots(num_words + 1, 1, figsize=(15, 5 * (num_words + 1)))

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

        # 使用getAttMap生成叠加图像
        gradcam_image = getAttMap(rgb_image, gradcam_map[i + 1])

        # 显示结果
        axes[i + 1].imshow(gradcam_image)
        axes[i + 1].set_title(f'Token: "{token}"')
        axes[i + 1].set_ylabel(token)
        axes[i + 1].set_yticks([])
        axes[i + 1].set_xticks([])

    # 设置总标题
    fig.suptitle(f'Caption: {caption}', fontsize=16, y=0.95)
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved word attention visualization to: {save_path}")
    return fig


def create_single_image_dataset(image_id, image_dir, annotation_path, rpn_file, tokenizer):
    """为单个图像创建数据集样本"""
    # 加载注解数据
    annotation_df = load_flickr_annotations(annotation_path)

    # 筛选指定图像ID的注解
    image_annotations = annotation_df[annotation_df['image_id'] == image_id]

    if len(image_annotations) == 0:
        raise ValueError(f"Image ID {image_id} not found in annotations")

    # 创建单样本数据集
    class SingleImageDataset:
        def __init__(self, image_id, image_dir, annotations, rpn_file, tokenizer):
            self.image_id = image_id
            self.image_dir = image_dir
            self.annotations = annotations
            self.tokenizer = tokenizer
            self.rpn_proposals = None
            self.num_crops = 32
            self.rpn_ratio = 0.5

            # 加载RPN建议框
            if rpn_file and os.path.exists(rpn_file):
                with open(rpn_file, 'r') as f:
                    self.rpn_proposals = json.load(f)

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            img_path = os.path.join(self.image_dir, self.image_id)

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
                    proposals = self.rpn_proposals.get(self.image_id.split('.')[0], [])
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

                # 处理文本
                captions = self.annotations['caption'].tolist()
                selected_captions = [captions[-1]]  # 使用第一个字幕
                original_text = captions[-1]

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

                return {
                    'images': images,
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'image_id': self.image_id,
                    'caption_ids': [0],
                    'original_text': original_text
                }

            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                return None

    return SingleImageDataset(image_id, image_dir, image_annotations, rpn_file, tokenizer)


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
    target_image_id = '7739176066.jpg'

    # 加载模型
    model = load_trained_model(model_path, device)

    # 临时设置BatchNorm使用运行统计量
    set_bn_training_mode(model, training=True, use_running_stats=True)

    # 创建输出目录
    os.makedirs("gradcam_results", exist_ok=True)

    # 加载tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained('./my_bert')
    except:
        print("Warning: Could not load tokenizer, using simple word splitting")
        tokenizer = None

    # 创建单图像数据集
    dataset = create_single_image_dataset(target_image_id, image_dir, annotation_path, rpn_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    target_layer = 'cross_attention'
    gradcam = GradCAM(model, target_layer)

    try:
        # 获取数据批次
        for batch in dataloader:
            if batch is None:
                continue

            images = batch['images'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            original_text = batch['original_text']  # 获取原始文本

            # 获取有效token的索引
            valid_indices, valid_tokens = get_valid_tokens_indices(input_ids.cpu(), tokenizer)

            print(f"Valid tokens: {valid_tokens}")
            print(f"Valid indices: {valid_indices}")
            print(f"Total tokens: {len(input_ids[0][0])}, Valid tokens: {len(valid_tokens)}")
            print(f"Image shape: {images.shape}")

            # 前向传播
            model.zero_grad()
            outputs = model(images, input_ids, attention_mask, original_text)
            loss = model.compute_loss(outputs)
            print("loss: ", loss)

            # 反向传播获取梯度
            loss.backward(retain_graph=True)

            # 获取梯度信息
            gradients = gradcam.get_gradients()
            print(f"Gradients stats - max: {gradients.max().item() if gradients is not None else 'None'}, "
                  f"min: {gradients.min().item() if gradients is not None else 'None'}, "
                  f"mean: {gradients.mean().item() if gradients is not None else 'None'}")

            print(f"Processing image: {target_image_id}")
            print(f"Caption: {original_text}")

            # 使用compute_gradcam函数计算Grad-CAM
            if hasattr(model, 'compute_gradcam'):
                # 提取图像特征
                global_feat, x1 = model.global_path(images[:, 0])
                x1 = model.image_attention_proj(x1)

                # 计算Grad-CAM
                gradcam_map = model.compute_gradcam(
                    x1,
                    gradcam,
                    attention_mask=attention_mask.squeeze(1),
                    target_token_idx=None
                )

                if gradcam_map is not None:
                    # 可视化每个单词的注意力
                    save_path = f"gradcam_results/sample_{target_image_id.replace('.', '_')}_word_attention.png"

                    # 将gradcam对象传递给visualize_word_attention函数
                    fig = visualize_word_attention(
                        images, gradcam_map, valid_tokens, valid_indices,
                        os.path.join(image_dir, target_image_id), original_text, save_path
                    )

                    print(f"Word attention visualization saved to: {save_path}")

            # 如果compute_gradcam不可用，使用备选方法
            elif gradcam.activations is not None:
                attention_weights = gradcam.activations

                save_path = f"gradcam_results/sample_{target_image_id.replace('.', '_')}_word_attention.png"
                fig = visualize_word_attention(
                    images, attention_weights, valid_tokens, valid_indices,
                    os.path.join(image_dir, target_image_id), original_text, save_path
                )

            break  # 只处理一个样本

    except Exception as e:
        print(f"Error processing sample: {e}")
        import traceback
        traceback.print_exc()

    print("Grad-CAM visualization completed!")


if __name__ == "__main__":
    main()

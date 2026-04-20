import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import json
import pandas as pd
from torch.utils.data import DataLoader
from scipy.ndimage import filters
from transformers import BertModel, BertConfig, BertTokenizer
import torchvision.transforms as transforms
from PIL import Image

# 导入必要的模块（根据您的实际文件结构调整路径）
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


def find_target_token(caption, tokenizer, keyword=None):
    """找到目标token的索引"""
    if keyword is None:
        # 如果没有指定关键词，选择名词或动词
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(caption)
            # 优先选择名词和动词
            for token in doc:
                # if token.pos_ in ['NOUN', 'VERB'] and len(token.text) > 2:
                if token.pos_ in ['NOUN',] and len(token.text) > 2:
                    return token.text, None
            # 如果没有找到，返回第一个实词
            for token in doc:
                if not token.is_stop and not token.is_punct and len(token.text) > 2:
                    return token.text, None
        except:
            pass

        # 如果spacy不可用，选择最长的单词
        words = caption.split()
        if words:
            target_word = max(words, key=len)
            return target_word, None
        return None, None

    # 如果指定了关键词，找到对应的token索引
    tokens = tokenizer.tokenize(caption)
    if keyword in tokens:
        token_idx = tokens.index(keyword) + 1  # +1因为第一个token是[CLS]
        return keyword, token_idx
    else:
        # 尝试找到包含关键词的token
        for i, token in enumerate(tokens):
            if keyword.lower() in token.lower():
                return token, i + 1
        return keyword, None


def main():
    # 设备配置
    from latest_utils import set_seed
    # set_seed(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 模型路径（根据实际训练结果调整）
    model_path = "retriever_epoch_best3.pth"  # 替换为您的模型路径

    # 数据路径
    annotation_path = "flickr30k/results_20130124.token"
    image_dir = "flickr30k/flickr30k-images"
    rpn_file = "flickr30k/flickr30k_rpn_proposals-U.json"

    # 加载模型
    model = load_trained_model(model_path, device)

    # 打印模型结构(部分)
    # print("\n[DEBUG] Model layers:")
    # for name, module in model.named_modules():
    #     if "cross_attention" in name:
    #         print(f" {name}:{module}")

    # 初始化可视化器
    # visualizer = GradCAMVisualizer(model, device)

    # 加载测试数据
    annotation_df = load_flickr_annotations(annotation_path)

    # 使用测试集或验证集（根据您的数据划分）
    image_ids = list(set(annotation_df['image_id']))
    test_image_ids = image_ids[:500]  # 使用前32张作为测试示例
    test_annotations = annotation_df[annotation_df['image_id'].isin(test_image_ids)]

    # 创建数据集
    eval_dataset = Flickr30kDataset(
        image_dir, test_annotations, './DATA/f30k_precomp',
        None, 'train', mode='train', rpn_proposals_file=rpn_file
    )

    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=True, num_workers=4)

    # 创建输出目录
    os.makedirs("gradcam_results", exist_ok=True)

    # 加载tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained('./my_bert')
    except:
        print("Warning: Could not load tokenizer, using simple word splitting")
        tokenizer = None

    # 可视化多个样本
    num_samples = 1  # 可视化的样本数量
    print(f"Visualizing {num_samples} samples...")

    target_layer = 'cross_attention'
    gradcam = GradCAM(model, target_layer)


    for i, batch in enumerate(eval_loader):
        if i >= num_samples:
            break

        try:
            # 获取数据
            images = batch['images'].to(device)
            texts = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            original_texts = batch['original_text']
            image_id = batch['image_id'][0]
            # caption_ids = batch['caption_ids'][0]

            # 前向传播
            model.zero_grad()
            outputs = model(images, texts, attention_mask, original_texts)
            loss = model.compute_loss(outputs)
            print("loss: ", loss)

            loss.backward(retain_graph=True)

            # 在反向传播后获取梯度
            gradients = gradcam.get_gradients()
            if gradients is not None:
                print(f"Gradients stats - max: {gradients.max().item() if gradients is not None else 'None'}, "
                      f"min: {gradients.min().item() if gradients is not None else 'None'}, "
                      f"mean: {gradients.mean().item() if gradients is not None else 'None'}")
            else:
                print("[DEBUG] No gradients available after backward")

            # 处理原始文本
            if isinstance(original_texts, list) and len(original_texts) > 0:
                if isinstance(original_texts[0], list):
                    caption = original_texts[0][0]  # 取第一个字幕
                else:
                    caption = original_texts[0]
            else:
                caption = "Unknown caption"

            print(f"Processing sample {i + 1}: {image_id}")
            print(f"Caption: {caption}")

            # 找到目标token
            target_word, target_token_idx = find_target_token(caption, tokenizer)
            print(f"Target word: {target_word}, Token index: {target_token_idx}")

            # 提取图像特征用于Grad-CAM计算
            # with torch.no_grad():
            global_feat, x1 = model.global_path(images[:, 0])
            global_text, local_text = model.text_encoder(texts, attention_mask, original_texts)
            local_text = local_text.squeeze(1)
            x1 = model.image_attention_proj(x1)

            # 计算Grad-CAM
            gradcam_map = model.compute_gradcam(
                x1,
                gradcam,
                attention_mask=attention_mask.squeeze(1),
                target_token_idx=target_token_idx
            )

            if gradcam_map is not None:
                # 加载原始图像
                img_path = os.path.join(image_dir, image_id)
                original_img = Image.open(img_path).convert('RGB')
                original_img = transform(original_img)

                # 可视化
                save_path = f"gradcam_results/sample_{i + 1}_{image_id.replace('.', '_')}"
                fig = model.visualize_gradcam(
                    original_img, gradcam_map, caption,
                    save_path=save_path
                )

                plt.close(fig)
                print(f"Successfully generated Grad-CAM for sample {i + 1}")
            else:
                print(f"Using fallback Grad-CAM method")
                # 使用回退方法
                gradcam_map = model.compute_alternative_gradcam(x1, target_token_idx)
                if gradcam_map is not None:
                    img_path = os.path.join(image_dir, image_id)
                    original_img = Image.open(img_path).convert('RGB')
                    save_path = f"gradcam_results/sample_{i + 1}_{image_id.replace('.', '_')}_fallback.png"
                    fig = model.visualize(original_img, gradcam_map[0], caption, save_path=save_path)
                    plt.close(fig)

        except Exception as e:
            print(f"Error processing sample {i + 1}: {e}")
            import traceback
            traceback.print_exc()
            continue

        print("Grad-CAM visualization completed!")
        exit(0)


if __name__ == "__main__":
    main()

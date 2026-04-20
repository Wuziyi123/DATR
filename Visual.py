import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from transformers import BertTokenizer
import cv2
import os
import sys
import clip
import pandas as pd
import random
from torch.utils.data import DataLoader, Dataset
import json

# 添加路径以便导入自定义模块
sys.path.append('./')

# 从您的代码中导入必要组件
try:
    from latest_utils import SimilarityComputer, SCAN_attention, l2norm
except ImportError:
    # 如果导入失败，定义这些函数
    def l2norm(X, dim, eps=1e-8):
        norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
        X = torch.div(X, norm)
        return X


    def SCAN_attention(query, context, smooth=9.0, eps=1e-8,):
        attn = torch.bmm(context, query.permute(0, 2, 1))
        attn = F.leaky_relu(attn, 0.1)
        attn = l2norm(attn, dim=2)

        attn = attn.permute(0, 2, 1)
        attn_weights = F.softmax(attn * smooth, dim=2)

        weighted_context = torch.bmm(attn_weights, context)
        weighted_context = l2norm(weighted_context, dim=-1)

        return weighted_context, attn_weights


# 加载训练数据集类（根据您的代码调整）
class Flickr30kDataset(Dataset):
    """Flickr30K数据集加载器（简化版）"""
    def __init__(self, annotation_df, image_dir, mode='train', num_samples=None):
        self.image_dir = image_dir
        self.mode = mode
        self.annotation_df = annotation_df

        # 按图像分组
        self.image_to_captions = annotation_df.groupby('image_id')['caption'].apply(list).to_dict()
        self.image_ids = list(self.image_to_captions.keys())

        # 如果指定了样本数量，随机选择一部分
        if num_samples and num_samples < len(self.image_ids):
            self.image_ids = random.sample(self.image_ids, num_samples)

        print(f"加载了 {len(self.image_ids)} 张图像用于可视化")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, image_id)
        captions = self.image_to_captions[image_id]

        # 随机选择一个描述（训练模式）或使用所有描述（评估模式）
        if self.mode == 'train':
            caption = random.choice(captions)
        else:
            caption = captions[0]  # 或者可以使用所有描述

        return {
            'image_path': image_path,
            'caption': caption,
            'captions': captions,  # 所有描述
            'image_id': image_id
        }


class AttentionVisualizer:
    """文本-图像匹配注意力可视化器 - 基于您的模型架构"""

    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self._load_model(model_path)
        self.tokenizer = BertTokenizer.from_pretrained('./my_bert')

        # 图像预处理（与训练时一致）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

        # 随机裁剪变换（用于局部图像）
        self.random_crop_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 0.9)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

    def _load_model(self, model_path):
        """加载训练好的模型"""
        # 首先加载CLIP模型（与训练时一致）
        clip_model, _ = clip.load("ViT-B/16", device=self.device)
        for param in clip_model.parameters():
            param.requires_grad = False

        trans_model, _ = clip.load("ViT-B/16", device=self.device)
        for param in trans_model.parameters():
            param.requires_grad = False

        # 初始化您的模型架构
        from latest import AdvancedCrossModalRetriever  # 从您的训练文件导入
        model = AdvancedCrossModalRetriever(clip_model, trans_model, num_crops=32)
        model = model.to(self.device)

        # 加载训练好的权重
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"模型加载成功，epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"最佳RSUM: {checkpoint.get('best_rsum', 'N/A')}")

        return model

    def preprocess_image(self, image_path, num_crops=32):
        """预处理图像 - 模拟训练时的处理流程"""
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)

        # 全局图像
        global_img = self.transform(image)

        # 生成局部裁剪（简化版，实际应该使用RPN提案）
        crop_imgs = []
        for _ in range(num_crops):
            crop_img = self.random_crop_transform(image)
            crop_imgs.append(crop_img)

        crop_imgs = torch.stack(crop_imgs)
        images = torch.cat([global_img.unsqueeze(0), crop_imgs], dim=0)

        return images.unsqueeze(0), original_image  # 添加batch维度

    def preprocess_text(self, text):
        """预处理文本"""
        text_input = self.tokenizer(
            text,
            padding='max_length',
            max_length=77,
            truncation=True,
            return_tensors='pt'
        )
        return text_input

    def extract_attention_weights(self, image_tensor, text_input):
        """提取模型中的注意力权重"""
        with torch.no_grad():
            # 获取模型输出
            outputs = self.model(
                image_tensor,
                text_input['input_ids'],
                text_input['attention_mask']
            )

            # 获取局部特征
            local_vis = outputs["local_vis"]  # [B, num_crops, D]
            local_text = outputs["local_text"]  # [B, seq_len, D]

            # 使用SCAN注意力计算权重
            attn_weights = SCAN_attention(
                local_text,  # query: 文本特征
                local_vis,  # context: 视觉特征
            )

            return attn_weights.squeeze(0).cpu().numpy(), local_vis.squeeze(0).cpu().numpy()

    def compute_similarity_map(self, image_path, text):
        """计算文本与图像区域的相似度图"""
        # 预处理
        image_tensor, original_image = self.preprocess_image(image_path)
        text_input = self.preprocess_text(text)

        # 移动到设备
        image_tensor = image_tensor.to(self.device)
        input_ids = text_input['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = text_input['attention_mask'].unsqueeze(0).to(self.device)

        # 提取注意力权重
        attn_weights, local_vis_features = self.extract_attention_weights(
            image_tensor,
            {'input_ids': input_ids, 'attention_mask': attention_mask}
        )

        # 注意力权重形状: [seq_len, num_crops]
        # 我们对每个文本token的注意力取平均，得到每个图像区域的总体注意力分数
        region_scores = attn_weights.mean(axis=0)  # [num_crops]

        return region_scores, original_image, local_vis_features

    def create_attention_overlay(self, original_image, region_scores, crop_positions=None):
        """创建注意力叠加图"""
        h, w = original_image.shape[:2]

        # 如果没有提供裁剪位置，假设均匀分布（简化处理）
        if crop_positions is None:
            # 创建基础注意力图
            base_attention = np.zeros((h, w))

            # 简化的注意力分布（实际应该根据RPN提案位置）
            # 这里我们创建一个中心加权的注意力图作为示例
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h / 2, w / 2
            dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
            base_attention = np.maximum(0, 1 - dist / max_dist)

            # 使用区域分数调整注意力
            overall_score = region_scores.mean() if len(region_scores) > 0 else 0.5
            attention_map = base_attention * overall_score
        else:
            # 如果有实际的裁剪位置，可以更精确地映射
            attention_map = np.zeros((h, w))
            for i, (x1, y1, x2, y2) in enumerate(crop_positions):
                if i < len(region_scores):
                    # 在裁剪区域填充注意力分数
                    attention_map[y1:y2, x1:x2] = region_scores[i]

        # 归一化注意力图
        if attention_map.max() > 0:
            attention_map = attention_map / attention_map.max()

        # 调整大小确保匹配
        attention_map = cv2.resize(attention_map, (w, h))

        # 创建热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 叠加到原图
        overlayed_image = cv2.addWeighted(original_image, 0.6, heatmap_rgb, 0.4, 0)

        return attention_map, overlayed_image

    def visualize(self, image_path, text, save_path=None, show_crops=False):
        """完整的可视化函数"""
        # 计算相似度图
        region_scores, original_image, local_features = self.compute_similarity_map(image_path, text)

        # 创建注意力叠加
        attention_map, overlayed_image = self.create_attention_overlay(original_image, region_scores)

        # 可视化设置
        fig = plt.figure(figsize=(20, 5 if not show_crops else 15))

        # 1. 原始图像
        plt.subplot(1, 4, 1)
        plt.imshow(original_image)
        plt.title('原始图像')
        plt.axis('off')

        # 2. 注意力热力图
        plt.subplot(1, 4, 2)
        plt.imshow(attention_map, cmap='hot')
        plt.title('注意力热力图')
        plt.axis('off')
        plt.colorbar()

        # 3. 叠加效果图
        plt.subplot(1, 4, 3)
        plt.imshow(overlayed_image)
        plt.title('文本: ' + (text[:40] + '...' if len(text) > 40 else text))
        plt.axis('off')

        # 4. 特征空间可视化（可选）
        plt.subplot(1, 4, 4)
        if len(local_features) > 1:
            # 使用PCA或t-SNE降维显示特征分布
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA

            # 使用PCA降维到2D
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(local_features)

            plt.scatter(features_2d[:, 0], features_2d[:, 1],
                        c=region_scores[:len(features_2d)],
                        cmap='viridis', alpha=0.7)
            plt.colorbar(label='注意力分数')
            plt.title('局部特征分布 (PCA)')
            plt.xlabel('PC1')
            plt.ylabel('PC2')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"可视化结果已保存至: {save_path}")

        plt.show()

        # 打印匹配分数
        match_score = region_scores.mean()
        print(f"文本-图像匹配分数: {match_score:.4f}")

        return match_score, attention_map

    def visualize_from_dataset(self, dataset, num_samples=5, output_dir="attention_results"):
        """从数据集中获取样本并进行可视化"""
        os.makedirs(output_dir, exist_ok=True)

        # 创建数据加载器
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        results = []
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            image_path = batch['image_path'][0]
            caption = batch['caption'][0]
            image_id = batch['image_id'][0]

            if not os.path.exists(image_path):
                print(f"图像不存在: {image_path}")
                continue

            save_path = os.path.join(output_dir, f"attention_{image_id}_{i + 1}.png")
            try:
                score, attn_map = self.visualize(image_path, caption, save_path)
                results.append({
                    'image_id': image_id,
                    'image_path': image_path,
                    'caption': caption,
                    'score': score,
                    'visualization': save_path
                })
            except Exception as e:
                print(f"处理 {image_path} 时出错: {e}")

        # 生成结果报告
        self._generate_report(results, output_dir)
        return results

    def _generate_report(self, results, output_dir):
        """生成可视化结果报告"""
        report_path = os.path.join(output_dir, "visualization_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("文本-图像注意力可视化报告\n")
            f.write("=" * 50 + "\n\n")

            for i, result in enumerate(results):
                f.write(f"样本 {i + 1}:\n")
                f.write(f"  图像ID: {result['image_id']}\n")
                f.write(f"  图像路径: {result['image_path']}\n")
                f.write(f"  文本: {result['caption']}\n")
                f.write(f"  匹配分数: {result['score']:.4f}\n")
                f.write(f"  可视化文件: {result['visualization']}\n\n")

        print(f"报告已生成: {report_path}")


# 辅助函数：加载Flickr30k标注数据
def load_flickr_annotations(annotation_path):
    """加载Flickr30k标注文件"""
    annotations = pd.read_table(
        annotation_path,
        sep='\t',
        header=None,
        names=['image_caption_id', 'caption']
    )
    annotations[['image_id', 'caption_id']] = annotations['image_caption_id'].str.split('#', expand=True)
    annotations['caption_id'] = annotations['caption_id'].astype(int)
    return annotations[['image_id', 'caption_id', 'caption']]


# 主函数
def main():
    # 配置参数
    model_path = "retriever_epoch_best.pth"  # 替换为您的实际模型路径
    annotation_path = "flickr30k/results_20130124.token"  # 替换为您的标注文件路径
    image_dir = "flickr30k/flickr30k-images"  # 替换为您的图像目录
    output_dir = "attention_visualization_results"
    num_samples = 10  # 要可视化的样本数量

    # 加载标注数据
    print("加载标注数据...")
    annotations = load_flickr_annotations(annotation_path)

    # 创建数据集
    print("创建数据集...")
    dataset = Flickr30kDataset(annotations, image_dir, mode='train', num_samples=100)

    # 初始化可视化器
    print("初始化可视化器...")
    visualizer = AttentionVisualizer(model_path)

    # 从数据集中获取样本并进行可视化
    print("开始可视化...")
    results = visualizer.visualize_from_dataset(dataset, num_samples, output_dir)

    print(f"完成了 {len(results)} 个样本的可视化")


if __name__ == "__main__":
    main()
    # 配置参数
    model_path = "retriever_epoch_best.pth"  # 替换为您的模型路径
    image_path = "flickr30k/flickr30k-images/553918837.jpg"  # 替换为您的图像路径
    text = "A little girl laughing while going down a slide."  # 替换为您的文本

    # 初始化可视化器
    visualizer = GradCAMVisualizer(model_path)

    # 可视化单个词汇
    print("可视化单个词汇...")
    visualizer.visualize_word_attention(
        image_path,
        text,
        target_word="girl",  # 指定要可视化的词汇
        save_path="girl_attention.png"
    )

    # 可视化多个词汇
    print("\n可视化多个词汇...")
    results = visualizer.visualize_multiple_words(
        image_path,
        text,
        words_to_visualize=["girl", "slide", "going"],  # 指定要可视化的词汇列表
        save_dir="gradcam_results"
    )
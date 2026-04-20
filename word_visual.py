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
import json
import os
import torchvision.transforms as transforms

import clip
# 导入必要的模块
from latest_datasetloader import Flickr30kDataset, transform, load_flickr_annotations
from latest_utils import SCAN_attention, l2norm


class GradCAM_SCAN_Visualizer:
    def __init__(self, model_path, image_path, text, image_id, annotation_path, image_dir, rpn_file):
        self.model_path = model_path
        self.image_path = image_path
        self.text = text
        self.image_id = image_id
        self.annotation_path = annotation_path
        self.image_dir = image_dir
        self.rpn_file = rpn_file

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.attention_maps = {}
        self.gradients = {}

        # 图像预处理
        self.normalize = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711)
        )
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize,
        ])

    def load_model(self):
        """加载预训练模型"""
        print("Loading model...")

        # 实际使用时需要根据您的模型结构进行加载
        # 首先加载CLIP模型（与训练时一致）
        clip_model, _ = clip.load("ViT-B/16", device=self.device)
        for param in clip_model.parameters():
            param.requires_grad = False

        trans_model, _ = clip.load("ViT-B/16", device=self.device)
        for param in trans_model.parameters():
            param.requires_grad = False

        from latest import AdvancedCrossModalRetriever  # 从您的训练文件导入
        self.model  = AdvancedCrossModalRetriever(clip_model, trans_model, num_crops=32)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")

    def load_single_data(self):
        """加载单张图片和文本数据"""
        print("Loading data...")

        # 加载标注数据
        annotation_df = load_flickr_annotations(self.annotation_path)

        # 创建单样本数据集
        single_annotation = annotation_df[annotation_df['image_id'] == self.image_id]

        dataset = Flickr30kDataset(
            image_dir=self.image_dir,
            annotation_df=single_annotation,
            data_path='./DATA/f30k_precomp',
            vocab=None,
            data_split='train',
            mode='train',
            rpn_proposals_file=self.rpn_file
        )

        # 获取单个样本
        sample = dataset[0]

        # 处理图像数据
        image = Image.open(self.image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)

        # 处理文本数据
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('./my_bert')
        text_input = tokenizer(
            self.text,
            padding='max_length',
            max_length=77,
            truncation=True,
            return_tensors='pt'
        )

        return image_tensor, text_input, image

    def hook_scan_attention(self):
        """注册钩子来捕获SCAN attention层的输出和梯度"""

        def forward_hook(module, input, output):
            self.attention_maps['scan_attention'] = output[1]  # 假设返回(attention_output, attention_weights)

        def backward_hook(module, grad_input, grad_output):
            self.gradients['scan_attention'] = grad_output[0]

        # 找到SCAN attention层并注册钩子
        for name, module in self.model.named_modules():
            if 'scan_attention' in name.lower() or 'similarity_computer' in name:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                print(f"Registered hooks for: {name}")

    def compute_gradcam(self, image, text_input):
        """计算Grad-CAM"""
        # 前向传播
        outputs = self.model(image, text_input.input_ids, text_input.attention_mask)

        # 获取相似度分数作为目标
        target = outputs["sim_matrix"].diag().sum()  # 使用对角线元素的和作为目标

        # 反向传播
        self.model.zero_grad()
        target.backward(retain_graph=True)

        # 获取attention maps和gradients
        attention_maps = self.attention_maps['scan_attention']
        gradients = self.gradients['scan_attention']

        # 计算权重：对梯度进行全局平均池化
        weights = F.adaptive_avg_pool2d(gradients, (1, 1))
        weights = weights.squeeze(-1).squeeze(-1)

        # 计算加权attention map
        gradcam = torch.zeros_like(attention_maps)
        for i in range(weights.size(1)):
            gradcam += weights[:, i:i + 1, :, :] * attention_maps[:, i, :, :]

        # ReLU激活
        gradcam = F.relu(gradcam)

        return gradcam, attention_maps

    def get_att_map(self, img, attMap, blur=True, overlap=True):
        """生成注意力热力图（参考文档2）"""
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
            attMap = 1 * (1 - attMap ** 0.7).reshape(attMap.shape + (1,)) * img + \
                     (attMap ** 0.7).reshape(attMap.shape + (1,)) * attMapV

        return attMap

    def visualize_word_attention(self, gradcam, attention_maps, original_image, text_input, tokenizer):
        """可视化每个单词对应的注意力"""
        # 获取文本tokens
        tokens = tokenizer.convert_ids_to_tokens(text_input.input_ids[0])

        # 创建可视化图像
        rgb_image = np.array(original_image)
        rgb_image = np.float32(rgb_image) / 255

        # 为每个token创建热力图
        num_tokens = len(tokens)
        fig, axes = plt.subplots(1, num_tokens + 1, figsize=(20, 5))

        # 显示原始图像
        axes[0].imshow(rgb_image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # 为每个token显示注意力热力图
        for i, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue

            # 获取对应token的attention map
            token_attention = gradcam[0, i].cpu().detach().numpy()

            # 生成热力图
            heatmap = self.get_att_map(rgb_image, token_attention)

            axes[i + 1].imshow(heatmap)
            axes[i + 1].set_title(f"Token: {token}")
            axes[i + 1].axis('off')

        plt.tight_layout()
        plt.show()

        return fig

    def process(self):
        """主处理流程"""
        try:
            # 1. 加载模型
            self.load_model()

            # 2. 注册钩子
            self.hook_scan_attention()

            # 3. 加载数据
            image_tensor, text_input, original_image = self.load_single_data()
            image_tensor = image_tensor.to(self.device)
            text_input = text_input.to(self.device)

            # 4. 计算Grad-CAM
            gradcam, attention_maps = self.compute_gradcam(image_tensor, text_input)

            # 5. 可视化
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('./my_bert')
            self.visualize_word_attention(gradcam, attention_maps, original_image, text_input, tokenizer)

            print("Visualization completed successfully!")

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            import traceback
            traceback.print_exc()


# 自定义SCAN attention层用于捕获中间结果
class SCANAttentionWrapper(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, query, context, smooth=9.0):
        """包装SCAN attention以捕获中间结果"""
        # 计算注意力分数
        attn = torch.bmm(context, query.permute(0, 2, 1))
        attn = F.leaky_relu(attn, 0.1)
        attn = l2norm(attn, dim=2)

        # 应用softmax获取注意力权重
        attn = attn.permute(0, 2, 1)
        attention_weights = F.softmax(attn * smooth, dim=2)

        # 计算加权上下文
        weighted_context = torch.bmm(attention_weights, context)
        weighted_context = l2norm(weighted_context, dim=-1)

        return weighted_context, attention_weights


def main():
    # 配置参数
    model_path = "retriever_epoch_best.pth"
    image_path = "flickr30k/flickr30k-images/553918837.jpg"
    text = "A little girl laughing while going down a slide"
    image_id = "553918837"
    annotation_path = "flickr30k/results_20130124.token"
    image_dir = "flickr30k/flickr30k-images"
    rpn_file = "flickr30k/flickr30k_rpn_proposals-U.json"

    # 创建可视化器
    visualizer = GradCAM_SCAN_Visualizer(
        model_path=model_path,
        image_path=image_path,
        text=text,
        image_id=image_id,
        annotation_path=annotation_path,
        image_dir=image_dir,
        rpn_file=rpn_file
    )

    # 执行可视化
    visualizer.process()


if __name__ == "__main__":
    main()
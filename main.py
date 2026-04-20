import json
import os
import pickle
import fire
import numpy as np
import torch
import yaml
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from flickr30k import load_flickr_annotations
from helper import set_seed, load_precomputed_features
from clip import clip
from torchvision import datasets
from typing import Tuple, List
from torch.nn import functional as F
from PIL import Image
from clip.simple_tokenizer import SimpleTokenizer

# 全局标注字典
annotation_dict = {}


def safe_tokenize(text, context_length=77):
    """使用官方tokenize并处理异常"""
    try:
        return clip.tokenize(text, truncate=True)[0]
    except RuntimeError:
        tokenizer = SimpleTokenizer()
        tokens = tokenizer.encode(text)
        tokens = tokens[:context_length - 2]
        return torch.tensor(
            [tokenizer.encoder["<|startoftext|>"]] +
            tokens +
            [tokenizer.encoder["<|endoftext|>"]],
            dtype=torch.long
        )


class Flickr30kDataset(Dataset):
    """Flickr30k数据集加载器"""

    def __init__(self, image_dir, annotation_df, num_crops=5):
        self.image_dir = image_dir
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)
                                   if f.endswith('.jpg')])
        self.captions = []
        self.annotation_df = annotation_df

        # 构建图像ID到描述的映射
        for img_id in annotation_df['image_id'].unique():
            img_captions = annotation_df[annotation_df['image_id'] == img_id]['caption'].tolist()[:5]
            self.captions.append(img_captions)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = datasets.folder.default_loader(img_path)
        img_id = os.path.basename(img_path).split('.')[0]

        # 获取对应的5个描述
        captions = self.captions[idx]
        return img, captions


def load_trained_model(
        model,
        dataset_name: str,
        model_size: str,
        alpha: float,
        n_samples: int,
        batch_size: int,
        num_workers: int,
        data_path: str,
        custom_loader: callable,
        device: torch.device,
        processor
):
    """加载训练好的模型"""
    save_file = (dataset_name + "-" + model_size).replace("/", "-")
    save_root = f"/home/wuziyi/Project/WCA/weights/{dataset_name}"
    model_save_path = os.path.join(save_root, f"{save_file}-{alpha}-{n_samples}.pkl")

    # 如果模型文件存在，直接加载
    if os.path.exists(model_save_path):
        print(f"Loading trained model from {model_save_path}...")
        from helper import GOAL_CLIP  # 导入您的模型类
        trained_model = GOAL_CLIP(clip_model=model, device=device)
        trained_model.load_state_dict(torch.load(model_save_path))
        trained_model.to(device).eval()
        return trained_model
    else:
        # 否则训练新模型
        print(f"Model not found, training new model...")
        load_precomputed_features(
            model,
            dataset_name=dataset_name,
            model_size=model_size,
            alpha=alpha,
            n_samples=n_samples,
            batch_size=batch_size,
            num_workers=num_workers,
            data_path=data_path,
            custom_loader=custom_loader,
            device=device,
            processor=processor,
        )
        exit(0)


def main(
        dataset_name: str = "flickr30k",
        num_workers: int = 16,
        seed: int = 0,
        device: str = "cuda",
):
    device = torch.device(device)
    print("Device:", device)
    print("num_workers:", num_workers)

    # 加载Flickr30k配置文件
    with open(file=f"cfgs/{dataset_name}.yaml") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
    set_seed(seed)

    # 加载超参数
    model_size = hparams["model_size"]
    alpha = hparams["alpha"]
    n_samples = hparams["n_samples"]
    batch_size = hparams["batch_size"]
    data_path = hparams["data_path"]

    global annotation_dict
    annotation_path = '/home/wuziyi/Project/Flickr/Flickr30K_Entities/data/results_20130124.token'
    annotations_df = load_flickr_annotations(annotation_path)
    for img_id, group in annotations_df.groupby('image_id'):
        annotation_dict[img_id] = group['caption'].tolist()[:5]

    # 加载CLIP模型
    print(f"Loading {model_size}")
    clip_model, processor = clip.load(model_size, device=device)
    clip_model.eval()
    clip_model.requires_grad_(False)

    # 加载训练好的模型
    trained_model = load_trained_model(
        clip_model,
        dataset_name=dataset_name,
        model_size=model_size,
        alpha=alpha,
        n_samples=n_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        data_path=data_path,
        custom_loader=None,
        device=device,
        processor=processor
    )

    # 创建数据集
    image_dir = '/home/wuziyi/Project/Flickr/Flickr30K_Entities/data/flickr30k-images'
    dataset = Flickr30kDataset(image_dir=image_dir, annotation_df=annotations_df)
    image_paths = dataset.image_paths
    all_captions = dataset.captions

    # ===== 只评估5%的数据集 =====
    total_samples = len(image_paths)
    sample_size = int(total_samples * 0.05)  # 5%的样本

    # 创建随机索引
    torch.manual_seed(seed)
    indices = torch.randperm(total_samples)[:sample_size]

    # 采样数据
    image_paths = [image_paths[i] for i in indices]
    all_captions = [all_captions[i] for i in indices]

    print(f"评估数据集大小: {len(image_paths)}/{total_samples} (5%)")

    # 创建评估数据加载器
    eval_dataset = [(image_paths[i], all_captions[i]) for i in range(len(image_paths))]
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: [
            torch.stack([processor(datasets.folder.default_loader(item[0])) for item in batch]),
            [item[1] for item in batch]
        ]
    )

    # Flickr30k评估指标
    def recall_at_k(sim_matrix: torch.Tensor, k_vals: tuple = (1, 5, 10)) -> dict:
        results = {}
        n_images = sim_matrix.size(0)

        # 图像->文本召回率
        img2txt_recall = {k: 0 for k in k_vals}
        for i in range(n_images):
            _, topk_indices = sim_matrix[i].topk(max(k_vals))
            for k in k_vals:
                if any(idx in range(i * 5, i * 5 + 5) for idx in topk_indices[:k]):
                    img2txt_recall[k] += 1
        for k in k_vals:
            img2txt_recall[k] = img2txt_recall[k] / n_images * 100

        # 文本->图像召回率
        txt2img_recall = {k: 0 for k in k_vals}
        sim_matrix_t = sim_matrix.t()
        for j in range(sim_matrix_t.size(0)):
            _, topk_indices = sim_matrix_t[j].topk(max(k_vals))
            img_idx = j // 5
            for k in k_vals:
                if img_idx in topk_indices[:k]:
                    txt2img_recall[k] += 1
        for k in k_vals:
            txt2img_recall[k] = txt2img_recall[k] / sim_matrix_t.size(0) * 100

        return {
            "image_to_text": img2txt_recall,
            "text_to_image": txt2img_recall
        }

    def dtype():
        return trained_model.visual.conv1.weight.dtype

    # 评估逻辑
    results = {}
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            methods = hparams["methods"]
            for method in methods:
                method = list(method.values())[0]
                method_name = method["name"]
                method_enabled = method["enabled"]

                if not method_enabled:
                    continue

                all_image_features = []
                all_text_features = []

                for images, captions_list in tqdm(eval_loader):
                    images = images.to(device)

                    # 提取特征
                    if method_name == "baseline":
                        # 使用原始CLIP模型
                        image_features = clip_model.encode_image(images)
                        text_features = []
                        for captions in captions_list:
                            text_inputs = torch.stack([safe_tokenize(c) for c in captions]).to(device)
                            text_feats = clip_model.encode_text(text_inputs)
                            text_features.append(text_feats)
                        text_features = torch.cat(text_features)
                    elif method_name == "ours":
                        # 使用训练好的模型
                        image_features = trained_model.visual(images.type(dtype()))
                        text_features = []
                        for captions in captions_list:
                            text_inputs = torch.stack([safe_tokenize(c) for c in captions]).to(device)
                            text_feats = trained_model.encode_text(text_inputs)
                            text_features.append(text_feats)
                        text_features = torch.cat(text_features)

                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())

                image_features = torch.cat(all_image_features)
                text_features = torch.cat(all_text_features)

                # 计算相似度矩阵
                if method_name == "baseline":
                    sim_matrix = image_features.float() @ text_features.t().float()
                elif method_name == "ours":
                    acc_list = []
                    patch_num = hparams["patch_n"]
                    if image_features.dim() == 2:
                        # 将二维特征转换为三维 (batch_size, 1, feature_dim)
                        image_features = image_features.unsqueeze(1)

                    for i in range(hparams["n_run"]):
                        # 采样局部特征 - 确保在有效范围内采样
                        num_patches = image_features.size(1)
                        random_indices = torch.randint(0, num_patches, (patch_num,))
                        patch_embeds = image_features[:, random_indices, :]

                        # 加权融合
                        weights = torch.softmax(torch.randn(patch_num), dim=0).to(patch_embeds.device)
                        weights = weights.view(1, -1, 1)

                        fused_feature = (patch_embeds * weights).sum(dim=1)

                        sim_matrix = fused_feature.float() @ text_features.t().float()
                        acc_list.append(recall_at_k(sim_matrix))

                    # 计算平均指标
                    avg_results = {}
                    for key in acc_list[0].keys():
                        avg_results[key] = {
                            k: np.mean([res[key][k] for res in acc_list])
                            for k in acc_list[0][key].keys()
                        }
                    results[method_name] = avg_results
                    continue

                # 评估检索性能
                results[method_name] = recall_at_k(sim_matrix)

    # 打印结果
    print("\nFlickr30k Evaluation Results:")
    for method, metrics in results.items():
        print(f"\n--- {method} ---")
        print("Image-to-Text Recall:")
        for k, v in metrics["image_to_text"].items():
            print(f"  R@{k}: {v:.2f}%")
        print("Text-to-Image Recall:")
        for k, v in metrics["text_to_image"].items():
            print(f"  R@{k}: {v:.2f}%")

    # 保存结果
    result_file = f"results/flickr30k_{model_size}_results.json"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {result_file}")


if __name__ == "__main__":
    fire.Fire(main)
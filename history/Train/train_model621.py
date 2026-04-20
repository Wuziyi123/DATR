import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import clip
from Flickr30k_RPN import Flickr30kDataset, load_flickr_annotations
from helper import GOAL_CLIP  # 假设您的模型定义在model.py中


def train_model():
    # 设置设备
    global epoch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据集参数
    ANNOTATION_PATH = "/home/wuziyi/Project/Flickr/Flickr30K_Entities/data/results_20130124.token"
    IMAGE_DIR = "/home/wuziyi/Project/Flickr/Flickr30K_Entities/data/flickr30k-images"
    proposal_file = '/home/wuziyi/Project/Flickr/Flickr30K_Entities/flickr30k_rpn_proposals-U.json'  # 预生成的提案文件

    NUM_CROPS = 12
    BATCH_SIZE = 16
    EPOCHS = 20
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.05

    # 创建数据集
    print("加载数据集...")
    annotations_df = load_flickr_annotations(ANNOTATION_PATH)

    # 拆分训练集/验证集 (80%/20%)
    image_ids = list(set(annotations_df['image_id']))
    # split_idx = int(len(image_ids) * 0.9)
    # split_idx = int(len(image_ids) * 0.95)
    # train_image_ids = image_ids[:split_idx]
    # train_image_ids = image_ids[:100]
    # val_image_ids = image_ids[split_idx:]

    # 新划分：1000测试集 + 1000评估集 + 其余训练集
    random.shuffle(image_ids)
    test_image_ids = image_ids[:1000]  # 前1000张测试集
    eval_image_ids = image_ids[1000:2000]  # 后续1000张评估集
    train_image_ids = image_ids[2000:]  # 其余为训练集

    # 创建训练集和验证集数据框
    train_annotations = annotations_df[annotations_df['image_id'].isin(train_image_ids)]
    val_annotations = annotations_df[annotations_df['image_id'].isin(val_image_ids)]

    # 训练和验证数据集
    train_dataset = Flickr30kDataset(
        image_dir=IMAGE_DIR,
        annotation_df=train_annotations,
        num_crops=NUM_CROPS,
        rpn_proposals_file=proposal_file,
        rpn_ratio=0.5,
    )

    val_dataset = Flickr30kDataset(
        image_dir=IMAGE_DIR,
        annotation_df=val_annotations,
        num_crops=NUM_CROPS,
    )

    # 自定义collate函数处理无效样本和文本填充
    def collate_fn(batch):
        # 过滤无效样本
        batch = [b for b in batch if b is not None and b[0] is not None]

        # 处理图像数据 [B, N_CROPS+1, C, H, W]
        images = torch.stack([item[0] for item in batch])

        # 处理文本数据 [B, 5, 77]
        texts = [item[1] for item in batch]
        # 已经由CLIP tokenizer处理为相同长度77，直接堆叠
        return images, torch.stack(texts)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print(f"训练集大小: {len(train_dataset)} | 验证集大小: {len(val_dataset)}")
    print(f"批大小: {BATCH_SIZE} | 每批步数: {len(train_loader)}")

    # 初始化CLIP模型
    clip_model, _ = clip.load("ViT-B/16", device=device, jit=False)

    # 初始化GOAL_CLIP模型
    model = GOAL_CLIP(
        clip_model=clip_model,
        device=device,
        num_local_experts=4,
        num_crops=NUM_CROPS
    ).to(device)

    # 初始化权重
    def init_weights(m):
        """特殊参数初始化"""
        if isinstance(m, nn.Linear):
            if 'rff' in m._get_name():
                nn.init.normal_(m.weight, mean=0, std=1 / m.weight.shape[1])
            elif 'expert' in m._get_name():
                nn.init.orthogonal_(m.weight)
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)

    # 设置优化器
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # 学习率调度器 (余弦退火)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * EPOCHS
    )

    # 混合精度训练的梯度缩放器
    scaler = GradScaler()

    # --------------------- 训练循环 ---------------------
    print("开始训练...")
    best_val_loss = float('inf')
    best_epoch = -1

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        train_steps = 0

        # 训练进度条
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"训练 Epoch {epoch + 1}/{EPOCHS}")

        for batch_idx, (images, texts) in train_bar:
            images = images.to(device, non_blocking=True)
            texts = texts.to(device, non_blocking=True)
            with autocast(enabled=True):
                loss = model(images, texts)
                # 反向传播
                scaler.scale(loss).backward()

            # 梯度裁剪
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # 更新学习率
            scheduler.step()

            # 记录损失
            total_train_loss += loss.item()
            train_steps += 1

            # 更新进度条
            if batch_idx % 50 == 0:
                train_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
                })

        # 计算平均训练损失
        avg_train_loss = total_train_loss / train_steps
        print(f"Epoch {epoch + 1}/{EPOCHS} | 训练损失: {avg_train_loss:.4f}")

        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        val_steps = 0

        # 初始化精度统计指标
        img2txt_top1 = 0.0
        img2txt_top5 = 0.0
        txt2img_top1 = 0.0
        txt2img_top5 = 0.0
        total_samples = 0

        # 验证进度条
        val_bar = tqdm(val_loader, desc=f"验证 Epoch {epoch + 1}/{EPOCHS}")

        with torch.no_grad():
            for images, texts in val_bar:
                images = images.to(device, non_blocking=True)
                texts = texts.to(device, non_blocking=True)
                B = images.size(0)  # 当前批次的图像数量

                with autocast():
                    # 前向传播获取logits
                    loss = model(images, texts)
                    total_val_loss += loss.item()
                    val_steps += 1

                    # 获取图像和文本特征
                    image_features = model.encode_image(images)  # [B, 512]
                    text_features = model.encode_text(texts)  # [B, 512]
                    # text_features = model.text_augmentation(text_features)

                    # 投影到共享空间
                    image_features = model.image_proj(image_features)
                    text_features = model.text_proj(text_features)

                    # 归一化特征
                    image_features = F.normalize(image_features, dim=-1)
                    text_features = F.normalize(text_features, dim=-1)

                    # 计算相似度矩阵（使用相同的logit_scale）
                    logit_scale = model.logit_scale.exp()
                    sim_matrix = logit_scale * image_features @ text_features.t()

                    # 计算图像到文本和文本到图像的检索精度
                    batch_size = image_features.size(0)
                    labels = torch.arange(batch_size, device=device)

                    # Image-to-Text (i2t) 检索
                    _, i2t_top1_indices = sim_matrix.topk(1, dim=1)
                    _, i2t_top5_indices = sim_matrix.topk(5, dim=1)

                    img2txt_top1 += (i2t_top1_indices.squeeze() == labels).sum().item()
                    img2txt_top5 += (torch.sum(i2t_top5_indices == labels.unsqueeze(1), dim=1)).sum().item()

                    # Text-to-Image (t2i) 检索
                    _, t2i_top1_indices = sim_matrix.topk(1, dim=0)
                    _, t2i_top5_indices = sim_matrix.topk(5, dim=0)

                    txt2img_top1 += (t2i_top1_indices.squeeze() == labels).sum().item()
                    txt2img_top5 += (torch.sum(t2i_top5_indices == labels.unsqueeze(0), dim=0)).sum().item()

                    total_samples += batch_size
                    total_val_loss += loss.item()
                    val_steps += 1
        # ====== 关键修正结束 ======

        # 计算平均损失
        avg_val_loss = total_val_loss / val_steps

        # 计算平均精度
        img2txt_top1 /= total_samples
        img2txt_top5 /= total_samples
        txt2img_top1 /= total_samples
        txt2img_top5 /= total_samples

        # 打印详细验证结果
        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"验证损失: {avg_val_loss:.4f} | "
              f"图像->文本 Top-1: {img2txt_top1:.4f} | "
              f"图像->文本 Top-5: {img2txt_top5:.4f} | "
              f"文本->图像 Top-1: {txt2img_top1:.4f} | "
              f"文本->图像 Top-5: {txt2img_top5:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            save_path = f"best_model_epoch{epoch + 1}_loss{avg_val_loss:.4f}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'img2txt_top1': img2txt_top1,
                'img2txt_top5': img2txt_top5,
                'txt2img_top1': txt2img_top1,
                'txt2img_top5': txt2img_top5,
            }, save_path)
            print(f"保存最佳模型到: {save_path}")
            print(f"当前最佳精度: 图像->文本 Top-1: {img2txt_top1:.4f}, 文本->图像 Top-1: {txt2img_top1:.4f}")

    # 保存最终模型
    final_save_path = f"save_weights/final_model_epochs{EPOCHS}_best{best_epoch + 1}.pth"
    if epoch % 3 == 0:
        torch.save(model.state_dict(), final_save_path)
        print(f"训练完成! 最终模型保存到: {final_save_path}")


if __name__ == "__main__":
    train_model()
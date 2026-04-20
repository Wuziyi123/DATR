import os
import random
from helper import set_seed

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
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
    set_seed(0)
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
    WEIGHT_DECAY = 0.05
    LEARNING_RATE = 1e-5  # 基础学习率
    FINAL_DECAY_LR = 5e-7  # 衰减终点学习率

    # 创建数据集
    print("加载数据集...")
    annotations_df = load_flickr_annotations(ANNOTATION_PATH)

    # 新划分：1000测试集 + 1000评估集 + 其余训练集
    image_ids = list(set(annotations_df['image_id']))
    random.shuffle(image_ids)
    test_image_ids = image_ids[60:90]  # 前1000张测试集
    eval_image_ids = image_ids[30:60]  # 后续1000张评估集
    train_image_ids = image_ids[0:30]  # 其余为训练集

    # 创建训练集和验证集数据框
    train_annotations = annotations_df[annotations_df['image_id'].isin(train_image_ids)]
    eval_annotations = annotations_df[annotations_df['image_id'].isin(eval_image_ids)]
    test_annotations = annotations_df[annotations_df['image_id'].isin(test_image_ids)]

    # 训练和验证数据集
    train_dataset = Flickr30kDataset(
        image_dir=IMAGE_DIR,
        annotation_df=train_annotations,
        num_crops=NUM_CROPS,
        rpn_proposals_file=proposal_file,
        rpn_ratio=0.5,
    )

    eval_dataset = Flickr30kDataset(
        image_dir=IMAGE_DIR,
        annotation_df=eval_annotations,
        num_crops=NUM_CROPS,
    )

    test_dataset = Flickr30kDataset(  # 新增测试集
        image_dir=IMAGE_DIR,
        annotation_df=test_annotations,
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

    eval_loader = DataLoader(  # 评估集加载器
        eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(  # 测试集加载器
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print(f"训练集大小: {len(train_dataset)} | 评估集大小: {len(eval_dataset)} | 测试集大小: {len(test_dataset)}")
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

    # 设置优化器（注意：初始学习率设为1.0，实际学习率由调度器完全控制）
    optimizer = AdamW(
        model.parameters(),
        lr=1.0,  # 设为1.0，实际学习率由LambdaLR完全控制
        weight_decay=WEIGHT_DECAY
    )

    # 重新定义学习率调度策略
    total_steps = EPOCHS * len(train_loader)  # 总步数
    fixed_steps = 3 * len(train_loader)  # 前3个epoch（固定学习率阶段）
    decay_start_step = fixed_steps  # 第4个epoch开始衰减
    decay_end_step = (EPOCHS-2) * len(train_loader)  # 第10个epoch结束衰减

    # 学习率lambda函数（精确控制固定和衰减）
    def lr_lambda(current_step):
        # 固定阶段：前2个epoch，学习率固定为1e-5
        if current_step < fixed_steps:
            return LEARNING_RATE  # 返回1e-5
        # 衰减阶段：第3到第10个epoch，线性衰减到1e-6
        elif current_step < decay_end_step:
            decay_progress = (current_step - fixed_steps) / (decay_end_step - fixed_steps)
            return LEARNING_RATE - (LEARNING_RATE - FINAL_DECAY_LR) * decay_progress
        # 稳定阶段：第10个epoch之后，学习率固定为1e-6
        else:
            return FINAL_DECAY_LR

    # 创建LambdaLR调度器（每个batch更新）
    scheduler = lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda
    )
    # 混合精度训练的梯度缩放器
    scaler = GradScaler()

    # --------------------- 训练循环 ---------------------
    print("开始训练...")
    best_val_acc = 0.0  # 跟踪最佳精度
    best_epoch = -1
    global_step = 0  # 跟踪全局步数

    for epoch in range(EPOCHS):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        train_steps = 0

        # 训练进度条
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                         desc=f"训练 Epoch {epoch + 1}/{EPOCHS} LR:{optimizer.param_groups[0]['lr']:.2e}")

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
            global_step += 1
            current_lr = optimizer.param_groups[0]['lr']   # 获取当前学习率用于显示

            # 记录损失
            total_train_loss += loss.item()
            train_steps += 1

            # 更新进度条
            if batch_idx % 50 == 0:
                train_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'LR': f'{current_lr:.2e}'
                })

        # 计算平均训练损失
        avg_train_loss = total_train_loss / train_steps
        print(f"Epoch {epoch + 1}/{EPOCHS} | 训练损失: {avg_train_loss:.4f}")

        # 验证阶段
        model.eval()
        # 初始化精度统计指标
        img2txt_top1 = 0.0
        img2txt_top5 = 0.0
        txt2img_top1 = 0.0
        txt2img_top5 = 0.0
        total_samples = 0

        # 验证进度条
        eval_bar = tqdm(eval_loader, desc=f"评估 Epoch {epoch + 1}/{EPOCHS}")

        with torch.no_grad():
            for images, texts in eval_bar:
                images = images.to(device, non_blocking=True)
                texts = texts.to(device, non_blocking=True)
                B = images.size(0)  # 当前批次的图像数量

                with autocast():
                    # 获取图像和文本特征
                    image_features = model.encode_image(images)  # [B, 512]
                    text_features = model.encode_text(texts)  # [B, 512]

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
        # ====== 关键修正结束 ======

        # 综合精度 (用于模型选择)
        current_acc = (img2txt_top1 + txt2img_top1) / 2
        # 计算精度指标
        img2txt_top1 /= total_samples
        img2txt_top5 /= total_samples
        txt2img_top1 /= total_samples
        txt2img_top5 /= total_samples

        # 打印详细验证结果
        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"图像->文本 Top-1: {img2txt_top1:.4f} | "
              f"图像->文本 Top-5: {img2txt_top5:.4f} | "
              f"文本->图像 Top-1: {txt2img_top1:.4f} | "
              f"文本->图像 Top-5: {txt2img_top5:.4f} | "
              f"综合精度: {current_acc:.4f}")

        # 保存最佳模型
        if current_acc > best_val_acc:
            best_val_acc = current_acc
            best_epoch = epoch
            save_path = f"best_model_epoch{epoch + 1}_acc{current_acc:.4f}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'img2txt_top1': img2txt_top1,
                'img2txt_top5': img2txt_top5,
                'txt2img_top1': txt2img_top1,
                'txt2img_top5': txt2img_top5,
                'composite_acc': current_acc,
            }, save_path)
            print(f"保存最佳模型到: {save_path}")
            print(f"当前最佳精度: 图像->文本 Top-1: {img2txt_top1:.4f}, 文本->图像 Top-1: {txt2img_top1:.4f}")

    # === 最终测试 (使用测试集) ===
    print("\n===== 在测试集上评估最佳模型 =====")
    checkpoint = torch.load(f"best_model_epoch{best_epoch + 1}_acc{best_val_acc:.4f}.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_img2txt_top1 = 0.0
    test_txt2img_top1 = 0.0
    test_total_samples = 0

    test_bar = tqdm(test_loader, desc="测试最佳模型")

    with torch.no_grad():
        for images, texts in test_bar:
            images = images.to(device)
            texts = texts.to(device)
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            image_features = model.image_proj(image_features)
            text_features = model.text_proj(text_features)

            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            logit_scale = model.logit_scale.exp()
            sim_matrix = logit_scale * image_features @ text_features.t()

            batch_size = image_features.size(0)
            labels = torch.arange(batch_size, device=device)

            # Image-to-Text
            _, i2t_top1_indices = sim_matrix.topk(1, dim=1)
            test_img2txt_top1 += (i2t_top1_indices.squeeze() == labels).sum().item()

            # Text-to-Image
            _, t2i_top1_indices = sim_matrix.topk(1, dim=0)
            test_txt2img_top1 += (t2i_top1_indices.squeeze() == labels).sum().item()

            test_total_samples += batch_size

    test_img2txt_top1 /= test_total_samples
    test_txt2img_top1 /= test_total_samples

    print(f"测试集最终性能 | "
          f"图像->文本 Top-1: {test_img2txt_top1:.4f} | "
          f"文本->图像 Top-1: {test_txt2img_top1:.4f}")


if __name__ == "__main__":
    train_model()
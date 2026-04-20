import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import sys

# 添加必要的导入路径
sys.path.append('.')
from latest import load_trained_model, AdvancedCrossModalRetriever
from latest_datasetloader import Flickr30kDataset, load_flickr_annotations


def i2t_single_image(npts, sims, image_idx):
    """
    为单个图像计算检索排名（参考文档4的i2t函数）
    Args:
        npts: 图像数量
        sims: 相似度矩阵 (N, 5N)
        image_idx: 当前图像索引
    """
    # 获取当前图像的相似度行
    img_similarities = sims[image_idx]

    # 按相似度降序排列
    inds = np.argsort(img_similarities)[::-1]

    # 找到对应5个文本的位置
    rank = 1e20
    correct_ranks = []

    # 当前图像对应的5个文本的起始位置
    correct_start = 5 * image_idx
    correct_end = 5 * image_idx + 5

    for i in range(correct_start, correct_end):
        tmp = np.where(inds == i)[0][0]
        correct_ranks.append(tmp)
        if tmp < rank:
            rank = tmp

    return rank, correct_ranks, inds


def evaluate_retrieval_visualization(model, eval_loader, device, tokenizer, save_dir="./retrieval_results",
                                     num_images=10):
    """
    评估阶段可视化检索结果：显示每张图片的前5个匹配字幕并标注是否正确
    改进版：参考文档4的评估函数，使用分块计算相似度矩阵

    Args:
        model: 训练好的图文检索模型
        eval_loader: 评估数据加载器
        device: 设备
        tokenizer: 文本tokenizer
        save_dir: 结果保存目录
        num_images: 要可视化的图像数量
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 设置模型为评估模式
    model.eval()
    model.mode = "eval"

    # 初始化特征存储
    img_embs = []
    local_img_embs = []
    cap_embs = []
    local_cap_embs = []
    attention_masks = []
    all_image_ids = []
    all_original_texts = []
    all_batch_indices = []

    print("提取特征中...")

    # 收集所有特征（参考文档4的validate函数）
    batch_counter = 0
    for i, batch in enumerate(tqdm(eval_loader, desc="Extracting Features")):
        images = batch['images'].to(device)
        texts = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        original_texts = batch['original_text']
        image_ids = batch['image_id']

        # 获取特征
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images, texts, attn_mask, original_texts)

        global_vis = outputs["global_vis"]
        local_vis = outputs["local_vis"]
        text_feats = outputs["text_feats"]
        local_text = outputs["local_text"]

        # 存储特征
        img_embs.append(global_vis)
        local_img_embs.append(local_vis)
        cap_embs.append(text_feats)
        local_cap_embs.append(local_text)
        attention_masks.append(attn_mask)

        # 存储元数据
        original_texts_list = [list(column) for column in zip(*original_texts)]
        for j, img_id in enumerate(image_ids):
            all_image_ids.append(img_id)
            all_original_texts.append(original_texts_list[j] if isinstance(original_texts_list, list) else original_texts)
            all_batch_indices.append((batch_counter, j))

        batch_counter += 1

        # 限制处理的批次数量以节省时间
        if batch_counter >= 3:  # 只处理前3个批次
            break

    # 合并特征
    img_embs = torch.cat(img_embs, dim=0)
    local_img_embs = torch.cat(local_img_embs, dim=0)
    cap_embs = torch.cat(cap_embs, dim=0)
    cap_embs = cap_embs.view(-1, cap_embs.size(-1))
    local_cap_embs = torch.cat(local_cap_embs, dim=0)
    local_cap_embs = local_cap_embs.view(-1, local_cap_embs.size(-2), local_cap_embs.size(-1))
    attention_masks = torch.cat(attention_masks, dim=0)
    attention_masks = attention_masks.view(-1, attention_masks.size(-1))

    n_img = len(img_embs)
    print(f"总共处理 {n_img} 张图像")

    # 分块计算相似度矩阵
    from evaluate import shard_attn_scores
    print("计算相似度矩阵...")
    sims = shard_attn_scores(model, img_embs, local_img_embs, cap_embs, local_cap_embs,
                             attention_masks, device, shard_size=20)

    # 为每个图像存储检索结果
    retrieval_results = []

    # 限制可视化的图像数量
    num_images_to_show = min(num_images, n_img)

    for img_idx in range(num_images_to_show):
        print(f"\n=== 处理图像 {img_idx + 1}/{num_images_to_show} ===")
        print(f"图像ID: {all_image_ids[img_idx]}")

        # 使用评估逻辑计算排名
        rank, correct_ranks, sorted_indices = i2t_single_image(n_img, sims, img_idx)

        # 获取前5个匹配结果
        top5_indices = sorted_indices[:5]
        top5_scores = sims[img_idx][top5_indices]

        # 将扁平索引转换为[图像索引, 字幕索引]
        top5_image_indices = top5_indices // 5
        top5_caption_indices = top5_indices % 5

        # 存储当前图像的检索结果
        img_results = {
            'image_id': all_image_ids[img_idx],
            'image_idx': img_idx,
            'true_rank': rank,
            'top5_matches': []
        }

        print(f"前5个匹配结果:")
        for rank_pos, (img_idx_match, cap_idx, score) in enumerate(
                zip(top5_image_indices, top5_caption_indices, top5_scores)):

            # 检查是否是正确的匹配（匹配到同一张图像）
            is_correct = (img_idx_match == img_idx)

            # 获取对应的文本
            if img_idx_match < len(all_original_texts):
                match_texts = all_original_texts[img_idx_match]
                if isinstance(match_texts, list) and cap_idx < len(match_texts):
                    match_text = match_texts[cap_idx]
                else:
                    match_text = str(match_texts)
            else:
                match_text = "文本不可用"

            result_info = {
                'rank': rank_pos + 1,
                'matched_image_id': all_image_ids[img_idx_match] if img_idx_match < len(all_image_ids) else "未知",
                'caption_index': cap_idx,
                'similarity_score': score,
                'is_correct': is_correct,
                'caption_text': match_text
            }

            img_results['top5_matches'].append(result_info)

            # 打印结果
            status = "✓ 正确" if is_correct else "✗ 错误"
            print(f"第{rank_pos + 1}名: 分数={score:.4f} {status}")
            print(f"   匹配图像: {result_info['matched_image_id']}")
            print(f"   字幕: {match_text}")

        retrieval_results.append(img_results)

        # 可视化当前图像的检索结果
        visualize_single_image_retrieval(
            img_idx, eval_loader, retrieval_results[-1], save_dir
        )

    # 生成总体统计报告
    generate_summary_report(retrieval_results, save_dir)

    return retrieval_results, sims


def visualize_single_image_retrieval(img_idx, eval_loader, img_results, save_dir):
    """
    可视化单张图像的检索结果
    """
    # 获取原始图像（从数据加载器中获取）
    try:
        # 获取包含目标图像的批次
        target_batch_idx = img_idx // eval_loader.batch_size
        target_in_batch_idx = img_idx % eval_loader.batch_size

        # 重置数据加载器并找到目标批次
        eval_loader_iter = iter(eval_loader)
        for i in range(target_batch_idx + 1):
            batch = next(eval_loader_iter)
            if i == target_batch_idx:
                images = batch['images']
                if target_in_batch_idx < len(images):
                    original_image = denormalize_and_convert(images[target_in_batch_idx][0])  # 取全局图像
                    break
        else:
            # 如果没找到，使用默认图像
            original_image = create_default_image()
    except:
        # 如果获取图像失败，创建默认图像
        original_image = create_default_image()

    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"图像检索结果 - 图像ID: {img_results['image_id']}", fontsize=16, y=0.95)

    # 显示原图
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('查询图像', fontsize=12)
    axes[0, 0].axis('off')

    # 显示排名信息
    axes[0, 1].text(0.1, 0.9, f"真实排名: #{img_results['true_rank'] + 1}",
                    transform=axes[0, 1].transAxes, fontsize=14, weight='bold')
    axes[0, 1].text(0.1, 0.7, f"图像ID: {img_results['image_id']}",
                    transform=axes[0, 1].transAxes, fontsize=12)
    axes[0, 1].axis('off')

    # 显示前5个匹配结果
    for i, match in enumerate(img_results['top5_matches']):
        row = 1 if i < 3 else 0
        col = i % 3

        if row < 2 and col < 3:  # 确保在子图范围内
            ax = axes[row, col]

            # 显示文本（如果是正确匹配用绿色框，错误用红色框）
            color = 'green' if match['is_correct'] else 'red'
            status = "✓ 正确匹配" if match['is_correct'] else "✗ 错误匹配"

            ax.text(0.05, 0.95, f"排名: {match['rank']}", transform=ax.transAxes,
                    fontsize=12, weight='bold', color=color)
            ax.text(0.05, 0.85, status, transform=ax.transAxes,
                    fontsize=11, weight='bold', color=color)
            ax.text(0.05, 0.75, f"相似度: {match['similarity_score']:.4f}",
                    transform=ax.transAxes, fontsize=10)
            ax.text(0.05, 0.65, f"匹配图像: {match['matched_image_id']}",
                    transform=ax.transAxes, fontsize=9, wrap=True)

            # 显示字幕文本（适当截断）
            caption_text = match['caption_text']
            if len(caption_text) > 80:
                caption_text = caption_text[:80] + "..."

            # 分割长文本为多行
            words = caption_text.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + word) < 40:
                    current_line += word + " "
                else:
                    lines.append(current_line)
                    current_line = word + " "
            if current_line:
                lines.append(current_line)

            for j, line in enumerate(lines):
                ax.text(0.05, 0.55 - j * 0.08, line, transform=ax.transAxes,
                        fontsize=9, wrap=True)

            # 添加边框颜色表示正确/错误
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_facecolor('#f8f9fa')

    # 隐藏多余的子图
    for i in range(len(img_results['top5_matches']), 6):
        row = i // 3
        col = i % 3
        if row < 2 and col < 3:
            axes[row, col].axis('off')

    # 调整布局
    plt.tight_layout()

    # 保存图像
    save_path = os.path.join(save_dir, f"retrieval_{img_results['image_id']}_{img_idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"检索结果已保存: {save_path}")


def denormalize_and_convert(tensor_image):
    """
    反归一化并转换张量为PIL图像
    """
    # ImageNet归一化参数
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])

    # 反归一化
    tensor_image = tensor_image.clone()
    for t, m, s in zip(tensor_image, mean, std):
        t.mul_(s).add_(m)

    # 转换为numpy并调整范围
    image_np = tensor_image.cpu().numpy().transpose(1, 2, 0)
    image_np = np.clip(image_np, 0, 1)

    # 转换为PIL图像
    return Image.fromarray((image_np * 255).astype(np.uint8))


def create_default_image():
    """创建默认图像用于显示"""
    default_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
    return Image.fromarray(default_img)


def generate_summary_report(retrieval_results, save_dir):
    """
    生成检索结果的统计报告
    """
    print("\n" + "=" * 60)
    print("检索结果统计报告")
    print("=" * 60)

    total_images = len(retrieval_results)
    top1_correct = 0
    top5_correct = 0
    mean_reciprocal_rank = 0.0

    # 计算各项指标
    for result in retrieval_results:
        # 检查top1是否正确
        if result['top5_matches'][0]['is_correct']:
            top1_correct += 1

        # 检查top5中是否有正确结果
        top5_has_correct = any(match['is_correct'] for match in result['top5_matches'])
        if top5_has_correct:
            top5_correct += 1

        # 计算平均倒数排名 (MRR)
        for match in result['top5_matches']:
            if match['is_correct']:
                mean_reciprocal_rank += 1.0 / match['rank']
                break

    # 计算准确率
    top1_accuracy = top1_correct / total_images * 100
    top5_accuracy = top5_correct / total_images * 100
    mrr = mean_reciprocal_rank / total_images

    print(f"总图像数: {total_images}")
    print(f"Top-1 准确率: {top1_accuracy:.2f}% ({top1_correct}/{total_images})")
    print(f"Top-5 准确率: {top5_accuracy:.2f}% ({top5_correct}/{total_images})")
    print(f"平均倒数排名 (MRR): {mrr:.4f}")

    # 生成详细报告
    report_data = []
    for result in retrieval_results:
        image_id = result['image_id']

        for match in result['top5_matches']:
            report_data.append({
                'image_id': image_id,
                'rank': match['rank'],
                'matched_image_id': match['matched_image_id'],
                'similarity_score': match['similarity_score'],
                'is_correct': match['is_correct'],
                'caption_text': match['caption_text'],
                'true_rank': result['true_rank'] + 1  # 转换为1-based索引
            })

    # 保存为CSV文件
    df_report = pd.DataFrame(report_data)
    csv_path = os.path.join(save_dir, "retrieval_detailed_report.csv")
    df_report.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 保存汇总统计
    summary_data = {
        'total_images': [total_images],
        'top1_correct': [top1_correct],
        'top5_correct': [top5_correct],
        'top1_accuracy': [top1_accuracy],
        'top5_accuracy': [top5_accuracy],
        'mrr': [mrr]
    }

    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(save_dir, "retrieval_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    print(f"\n详细报告已保存: {csv_path}")
    print(f"汇总统计已保存: {summary_path}")

    # 生成可视化图表
    plt.figure(figsize=(12, 8))

    # 创建子图
    plt.subplot(2, 2, 1)
    categories = ['Top-1准确率', 'Top-5准确率']
    accuracies = [top1_accuracy, top5_accuracy]

    bars = plt.bar(categories, accuracies, color=['#ff6b6b', '#51cf66'])

    # 在柱状图上添加数值标签
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=12)

    plt.ylabel('准确率 (%)', fontsize=12)
    plt.title('图文检索性能评估', fontsize=14)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)

    # 排名分布图
    plt.subplot(2, 2, 2)
    ranks = [result['true_rank'] + 1 for result in retrieval_results]  # 转换为1-based
    rank_counts = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1

    plt.bar(rank_counts.keys(), rank_counts.values(), color='skyblue')
    plt.xlabel('真实排名', fontsize=12)
    plt.ylabel('图像数量', fontsize=12)
    plt.title('真实排名分布', fontsize=14)
    plt.grid(axis='y', alpha=0.3)

    # 正确/错误分布图
    plt.subplot(2, 2, 3)
    correct_counts = [0] * 5
    for result in retrieval_results:
        for match in result['top5_matches']:
            if match['is_correct']:
                correct_counts[match['rank'] - 1] += 1

    plt.bar(range(1, 6), correct_counts, color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
    plt.xlabel('排名位置', fontsize=12)
    plt.ylabel('正确匹配数量', fontsize=12)
    plt.title('各排名位置正确匹配数量', fontsize=14)
    plt.xticks(range(1, 6))

    # 相似度分布图
    plt.subplot(2, 2, 4)
    all_scores = []
    for result in retrieval_results:
        for match in result['top5_matches']:
            all_scores.append(match['similarity_score'])

    plt.hist(all_scores, bins=20, alpha=0.7, color='purple')
    plt.xlabel('相似度分数', fontsize=12)
    plt.ylabel('频次', fontsize=12)
    plt.title('相似度分数分布', fontsize=14)

    plt.tight_layout()

    chart_path = os.path.join(save_dir, "retrieval_performance_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"性能图表已保存: {chart_path}")


def main():
    """
    主函数：加载模型并运行评估可视化
    """
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载训练好的模型
    model_path = "retriever_epoch_best3.pth"  # 替换为你的模型路径
    model = load_trained_model(model_path, device)

    # 加载评估数据集
    annotation_path = "flickr30k/results_20130124.token"
    image_dir = "flickr30k/flickr30k-images"
    rpn_file = "flickr30k/flickr30k_rpn_proposals-U.json"

    annotation_df = load_flickr_annotations(annotation_path)
    image_ids = list(set(annotation_df['image_id']))
    eval_image_ids = image_ids[500:520]  # 使用20张图像进行评估

    eval_annotations = annotation_df[annotation_df['image_id'].isin(eval_image_ids)]
    eval_dataset = Flickr30kDataset(
        image_dir, eval_annotations, "./DATA/f30k_precomp",
        None, 'dev', mode='eval', rpn_proposals_file=rpn_file
    )

    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=2)

    print(f"评估数据集大小: {len(eval_dataset)}")

    # 运行评估可视化
    retrieval_results, similarity_matrix = evaluate_retrieval_visualization(
        model, eval_loader, device, None, num_images=10
    )

    print("\n评估完成！所有结果已保存到 './retrieval_results/' 目录")

    return retrieval_results, similarity_matrix


if __name__ == "__main__":
    retrieval_results, sim_matrix = main()

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
import matplotlib.patches as patches

# 添加必要的导入路径
sys.path.append('.')
from latest import load_trained_model, AdvancedCrossModalRetriever
from latest_datasetloader import Flickr30kDataset, load_flickr_annotations


def i2t_single_image(npts, sims, image_idx):
    """
    为单个图像计算检索排名
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
                                     num_images=20, visualize_count=6):
    """
    评估阶段可视化检索结果：显示每张图片的前3个匹配字幕并标注是否正确
    改进版：按照文档1的格式进行可视化

    Args:
        model: 训练好的图文检索模型
        eval_loader: 评估数据加载器
        device: 设备
        tokenizer: 文本tokenizer
        save_dir: 结果保存目录
        num_images: 要处理的图像数量
        visualize_count: 要可视化的图像数量（6张）
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
    all_images_tensor = []  # 存储图像张量用于可视化

    print("Extracting features...")

    # 收集所有特征
    batch_counter = 0
    for i, batch in enumerate(tqdm(eval_loader, desc="Extracting Features")):
        images = batch['images'].to(device)
        texts = batch['input_ids'].to(device)
        attn_mask = batch['attention_mask'].to(device)
        original_texts = batch['original_text']
        image_ids = batch['image_id']

        # 存储图像张量
        all_images_tensor.extend([img.cpu() for img in images])

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
            all_original_texts.append(
                original_texts_list[j] if isinstance(original_texts_list, list) else original_texts)

        batch_counter += 1

        # 限制处理的批次数量
        if batch_counter >= 3:  # 只处理前3个批次（约20张图像）
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
    print(f"Total images processed: {n_img}")

    # 分块计算相似度矩阵
    from evaluate import shard_attn_scores
    print("Calculating similarity matrix...")
    sims = shard_attn_scores(model, img_embs, local_img_embs, cap_embs, local_cap_embs,
                             attention_masks, device, shard_size=20)

    # 为每个图像存储检索结果
    retrieval_results = []

    # 处理所有图像
    for img_idx in range(n_img):
        # 使用评估逻辑计算排名
        rank, correct_ranks, sorted_indices = i2t_single_image(n_img, sims, img_idx)

        # 获取前3个匹配结果
        top3_indices = sorted_indices[:3]
        top3_scores = sims[img_idx][top3_indices]

        # 将扁平索引转换为[图像索引, 字幕索引]
        top3_image_indices = top3_indices // 5
        top3_caption_indices = top3_indices % 5

        # 存储当前图像的检索结果
        img_results = {
            'image_id': all_image_ids[img_idx],
            'image_idx': img_idx,
            'image_tensor': all_images_tensor[img_idx] if img_idx < len(all_images_tensor) else None,
            'true_rank': rank,
            'top3_matches': []
        }

        for rank_pos, (img_idx_match, cap_idx, score) in enumerate(
                zip(top3_image_indices, top3_caption_indices, top3_scores)):

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
                match_text = "Text not available"

            result_info = {
                'rank': rank_pos + 1,
                'matched_image_id': all_image_ids[img_idx_match] if img_idx_match < len(all_image_ids) else "Unknown",
                'caption_index': cap_idx,
                'similarity_score': score,
                'is_correct': is_correct,
                'caption_text': match_text
            }

            img_results['top3_matches'].append(result_info)

        retrieval_results.append(img_results)

    # 选择前6张图像进行可视化
    visualize_indices = list(range(min(visualize_count, len(retrieval_results))))
    visualization_results = [retrieval_results[i] for i in visualize_indices]

    # 创建整体可视化图
    create_overall_visualization(visualization_results, save_dir)

    # 生成总体统计报告
    generate_summary_report(retrieval_results, save_dir)

    return retrieval_results, sims


def create_overall_visualization(retrieval_results, save_dir):
    """
    创建整体可视化图，按照文档1的格式显示6张图像的检索结果
    """
    # 创建更大的图，确保所有内容都能清晰显示
    fig = plt.figure(figsize=(20, 12))

    # 设置全局字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10

    # 创建网格布局：2行，每行3个主图，每个主图内有3个子区域
    gs = plt.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.15)

    # 遍历6个结果
    for idx, result in enumerate(retrieval_results):
        row = idx // 3
        col = idx % 3

        # 创建主图的网格
        main_gs = gs[row, col].subgridspec(2, 2, hspace=0.1, wspace=0.1, height_ratios=[3, 1])

        # 第一行：图像区域（查询图像和注意力图）
        ax_query = fig.add_subplot(main_gs[0, 0])  # 查询图像
        ax_attention = fig.add_subplot(main_gs[0, 1])  # 注意力图

        # 第二行：文本区域（合并两列）
        ax_text = fig.add_subplot(main_gs[1, :])

        # 显示查询图像
        if result['image_tensor'] is not None:
            try:
                # 将图像转换为PIL格式并显示
                img = denormalize_and_convert(result['image_tensor'])

                # 在查询图像子图中显示
                ax_query.imshow(img)
                ax_query.set_title('Query Image', fontsize=12, weight='bold', pad=10)
                ax_query.axis('off')

                # 在注意力图子图中也显示原图（暂时用原图代替注意力图）
                ax_attention.imshow(img)
                ax_attention.set_title('Attention', fontsize=12, weight='bold', pad=10)
                ax_attention.axis('off')

            except Exception as e:
                print(f"Error displaying image {result['image_id']}: {e}")
                # 显示默认图像
                ax_query.text(0.5, 0.5, 'Image\nNot Available', ha='center', va='center',
                              fontsize=12, style='italic', transform=ax_query.transAxes)
                ax_query.set_title('Query Image', fontsize=12, weight='bold')
                ax_query.axis('off')

                ax_attention.text(0.5, 0.5, 'Attention\n(Image)', ha='center', va='center',
                                  fontsize=12, style='italic', transform=ax_attention.transAxes)
                ax_attention.set_title('Attention', fontsize=12, weight='bold')
                ax_attention.axis('off')
        else:
            # 如果没有图像张量
            ax_query.text(0.5, 0.5, 'Image\nNot Available', ha='center', va='center',
                          fontsize=12, style='italic', transform=ax_query.transAxes)
            ax_query.set_title('Query Image', fontsize=12, weight='bold')
            ax_query.axis('off')

            ax_attention.text(0.5, 0.5, 'Attention\n(Image)', ha='center', va='center',
                              fontsize=12, style='italic', transform=ax_attention.transAxes)
            ax_attention.set_title('Attention', fontsize=12, weight='bold')
            ax_attention.axis('off')

        # 在文本区域显示前3个匹配结果
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)
        ax_text.axis('off')

        # 显示三个匹配结果，水平排列
        text_width = 0.3  # 每个文本区域的宽度
        spacing = 0.05  # 文本区域之间的间距

        for i, match in enumerate(result['top3_matches']):
            # 计算文本区域的x位置
            x_pos = i * (text_width + spacing) + 0.05

            # 确定标记和颜色
            if match['is_correct']:
                marker = '✓'
                color = 'black'  # 正确匹配用黑色
                marker_color = 'green'  # 勾号用绿色
            else:
                marker = '✗'
                color = 'red'  # 错误匹配用红色
                marker_color = 'red'  # 叉号用红色

            # 显示排名和标记
            rank_text = f"{i + 1}: {marker}"
            ax_text.text(x_pos, 0.7, rank_text, ha='left', va='center',
                         fontsize=12, weight='bold', color=marker_color)

            # 显示字幕文本（不换行，截断过长的文本）
            caption = match['caption_text']
            # 截断过长的文本
            if len(caption) > 80:
                caption = caption[:77] + "..."

            # 显示字幕文本
            ax_text.text(x_pos, 0.3, caption, ha='left', va='center',
                         fontsize=10, color=color, wrap=False)

        # 设置子图标题（图像标签(a)-(f)）
        label = f"({chr(97 + idx)})"  # a, b, c, d, e, f
        ax_query.text(0.02, 0.98, label, ha='left', va='top',
                      transform=ax_query.transAxes, fontsize=14, weight='bold',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # 调整布局
    plt.tight_layout()

    # 添加整体标题
    fig.suptitle("Qualitative Results of Image-to-Text Retrieval",
                 fontsize=16, weight='bold', y=0.98)

    # 保存图像
    save_path = os.path.join(save_dir, "overall_retrieval_visualization.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Overall visualization saved: {save_path}")


def denormalize_and_convert(tensor_image):
    """
    反归一化并转换张量为PIL图像
    """
    # 如果是元组（包含多个图像），取第一个（全局图像）
    if isinstance(tensor_image, tuple):
        tensor_image = tensor_image[0]

    # 确保是3通道图像
    if tensor_image.dim() == 3 and tensor_image.size(0) == 3:
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
    else:
        # 如果不是3通道图像，创建默认图像
        default_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        return Image.fromarray(default_img)


def generate_summary_report(retrieval_results, save_dir):
    """
    生成检索结果的统计报告
    """
    print("\n" + "=" * 60)
    print("Retrieval Performance Summary")
    print("=" * 60)

    total_images = len(retrieval_results)
    top1_correct = 0
    top3_correct = 0
    mean_reciprocal_rank = 0.0

    # 计算各项指标
    for result in retrieval_results:
        # 检查top1是否正确
        if result['top3_matches'] and result['top3_matches'][0]['is_correct']:
            top1_correct += 1

        # 检查top3中是否有正确结果
        top3_has_correct = any(match['is_correct'] for match in result['top3_matches'])
        if top3_has_correct:
            top3_correct += 1

        # 计算平均倒数排名 (MRR)
        for match in result['top3_matches']:
            if match['is_correct']:
                mean_reciprocal_rank += 1.0 / match['rank']
                break

    # 计算准确率
    top1_accuracy = top1_correct / total_images * 100
    top3_accuracy = top3_correct / total_images * 100
    mrr = mean_reciprocal_rank / total_images

    print(f"Total images: {total_images}")
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}% ({top1_correct}/{total_images})")
    print(f"Top-3 Accuracy: {top3_accuracy:.2f}% ({top3_correct}/{total_images})")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

    # 生成详细报告
    report_data = []
    for result in retrieval_results:
        image_id = result['image_id']

        for match in result['top3_matches']:
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
        'top3_correct': [top3_correct],
        'top1_accuracy': [top1_accuracy],
        'top3_accuracy': [top3_accuracy],
        'mrr': [mrr]
    }

    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(save_dir, "retrieval_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    print(f"\nDetailed report saved: {csv_path}")
    print(f"Summary statistics saved: {summary_path}")


def main():
    """
    主函数：加载模型并运行评估可视化
    """
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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

    print(f"Evaluation dataset size: {len(eval_dataset)}")

    # 运行评估可视化
    retrieval_results, similarity_matrix = evaluate_retrieval_visualization(
        model, eval_loader, device, None, num_images=20, visualize_count=6
    )

    print("\nEvaluation completed! All results saved to './retrieval_results/' directory")

    return retrieval_results, similarity_matrix


if __name__ == "__main__":
    retrieval_results, sim_matrix = main()

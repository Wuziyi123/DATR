import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd
from torch.utils.data import DataLoader


def evaluate_retrieval_visualization(model, eval_loader, device, tokenizer, save_dir="./retrieval_results"):
    """
    评估阶段可视化检索结果：显示每张图片的前5个匹配字幕并标注是否正确

    Args:
        model: 训练好的图文检索模型
        eval_loader: 评估数据加载器
        device: 设备
        tokenizer: 文本tokenizer
        save_dir: 结果保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 设置模型为评估模式
    model.eval()
    model.mode = "eval"

    # 获取一个批次的数据
    batch = next(iter(eval_loader))

    images = batch['images'].to(device, non_blocking=True)
    texts = batch['input_ids'].to(device, non_blocking=True)
    attention_mask = batch['attention_mask'].to(device, non_blocking=True)
    image_ids = batch['image_id']
    caption_ids = batch['caption_ids']
    original_texts = batch['original_text']

    batch_size = images.size(0)

    print(f"评估批次大小: {batch_size}")
    print(f"图像ID示例: {image_ids[:3]}")

    # 前向传播获取特征
    with torch.no_grad():
        outputs = model(images, texts, attention_mask, original_texts)

    # 提取特征
    image_features = outputs["global_vis"]  # [B, D]
    text_features = outputs["text_feats"]  # [B, 5, D] - 每个图像有5个候选字幕

    # 计算相似度矩阵
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)

    # 重塑文本特征: [B, 5, D] -> [B*5, D]
    text_features_flat = text_features.reshape(-1, text_features.size(-1))

    # 计算所有图像与所有文本的相似度 [B, B*5]
    similarity_matrix = torch.matmul(image_features, text_features_flat.t())

    # 重塑相似度矩阵以便理解: [B, B, 5]
    similarity_matrix_reshaped = similarity_matrix.reshape(batch_size, batch_size, 5)

    # 为每个图像存储检索结果
    retrieval_results = []

    for img_idx in range(batch_size):
        print(f"\n=== 图像 {img_idx + 1}/{batch_size} ===")
        print(f"图像ID: {image_ids[img_idx]}")

        # 获取当前图像的相似度分数 [B*5]
        img_similarities = similarity_matrix[img_idx]

        # 获取排序后的索引（从高到低）
        sorted_indices = torch.argsort(img_similarities, descending=True)

        # 获取前5个匹配结果
        top5_indices = sorted_indices[:5]
        top5_scores = img_similarities[top5_indices]

        # 将扁平索引转换为[图像索引, 字幕索引]
        top5_image_indices = top5_indices // 5
        top5_caption_indices = top5_indices % 5

        # 存储当前图像的检索结果
        img_results = {
            'image_id': image_ids[img_idx],
            'image_idx': img_idx,
            'top5_matches': []
        }

        print(f"前5个匹配结果:")
        for rank, (img_idx_match, cap_idx, score) in enumerate(
                zip(top5_image_indices, top5_caption_indices, top5_scores)):
            # 检查是否是正确的匹配（匹配到同一张图像）
            is_correct = (img_idx_match == img_idx)

            # 获取对应的文本
            match_text = original_texts[img_idx_match][cap_idx] if isinstance(original_texts[img_idx_match], list) else \
            original_texts[img_idx_match]

            result_info = {
                'rank': rank + 1,
                'matched_image_id': image_ids[img_idx_match],
                'caption_index': cap_idx.item(),
                'similarity_score': score.item(),
                'is_correct': is_correct,
                'caption_text': match_text
            }

            img_results['top5_matches'].append(result_info)

            # 打印结果
            status = "✓ 正确" if is_correct else "✗ 错误"
            print(f"第{rank + 1}名: 分数={score:.4f} {status}")
            print(f"   匹配图像: {image_ids[img_idx_match]}")
            print(f"   字幕: {match_text}")

        retrieval_results.append(img_results)

        # 可视化当前图像的检索结果
        visualize_single_image_retrieval(
            img_idx, batch, retrieval_results[-1], tokenizer, save_dir
        )

    # 生成总体统计报告
    generate_summary_report(retrieval_results, save_dir)

    return retrieval_results


def visualize_single_image_retrieval(img_idx, batch, img_results, tokenizer, save_dir):
    """
    可视化单张图像的检索结果
    """
    # 获取原始图像
    images = batch['images']
    original_image = denormalize_and_convert(images[img_idx][0])  # 取全局图像

    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"图像检索结果 - 图像ID: {img_results['image_id']}", fontsize=16, y=0.95)

    # 显示原图
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('查询图像', fontsize=12)
    axes[0, 0].axis('off')

    # 显示前5个匹配结果
    for i, match in enumerate(img_results['top5_matches']):
        row = (i + 1) // 3
        col = (i + 1) % 3

        if row < 2 and col < 3:  # 确保在子图范围内
            ax = axes[row, col]

            # 显示文本（如果是正确匹配用绿色框，错误用红色框）
            color = 'green' if match['is_correct'] else 'red'
            ax.text(0.1, 0.9, f"排名: {match['rank']}", transform=ax.transAxes,
                    fontsize=10, weight='bold', color=color)
            ax.text(0.1, 0.7, f"相似度: {match['similarity_score']:.4f}",
                    transform=ax.transAxes, fontsize=9)
            ax.text(0.1, 0.5, f"匹配图像: {match['matched_image_id']}",
                    transform=ax.transAxes, fontsize=8, wrap=True)

            # 显示字幕文本（适当截断）
            caption_text = match['caption_text']
            if len(caption_text) > 100:
                caption_text = caption_text[:100] + "..."

            ax.text(0.1, 0.3, f"字幕: {caption_text}",
                    transform=ax.transAxes, fontsize=8, wrap=True)

            # 添加边框颜色表示正确/错误
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_facecolor('#f8f9fa')

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

    # 计算各项指标
    for result in retrieval_results:
        # 检查top1是否正确
        if result['top5_matches'][0]['is_correct']:
            top1_correct += 1

        # 检查top5中是否有正确结果
        top5_has_correct = any(match['is_correct'] for match in result['top5_matches'])
        if top5_has_correct:
            top5_correct += 1

    # 计算准确率
    top1_accuracy = top1_correct / total_images * 100
    top5_accuracy = top5_correct / total_images * 100

    print(f"总图像数: {total_images}")
    print(f"Top-1 准确率: {top1_accuracy:.2f}% ({top1_correct}/{total_images})")
    print(f"Top-5 准确率: {top5_accuracy:.2f}% ({top5_correct}/{total_images})")

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
                'caption_text': match['caption_text']
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
        'top5_accuracy': [top5_accuracy]
    }

    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(save_dir, "retrieval_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    print(f"\n详细报告已保存: {csv_path}")
    print(f"汇总统计已保存: {summary_path}")

    # 生成可视化图表
    plt.figure(figsize=(10, 6))

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

    # 加载tokenizer（根据你的实际tokenizer调整）
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('./my_bert')

    # 加载评估数据集（使用文档2中的Flickr30kDataset）
    annotation_path = "flickr30k/results_20130124.token"
    image_dir = "flickr30k/flickr30k-images"
    rpn_file = "flickr30k/flickr30k_rpn_proposals-U.json"

    from latest_datasetloader import load_flickr_annotations, Flickr30kDataset

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
    retrieval_results = evaluate_retrieval_visualization(
        model, eval_loader, device, tokenizer
    )

    print("\n评估完成！所有结果已保存到 './retrieval_results/' 目录")


if __name__ == "__main__":
    # 添加必要的导入（根据你的实际代码调整）
    import torch.nn.functional as F
    from latest import load_trained_model  # 替换为你的实际导入路径

    main()

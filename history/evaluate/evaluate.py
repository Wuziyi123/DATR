import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import pywt
import numpy as np
from Flickr30k_RPN import Flickr30kDataset, load_flickr_annotations
from tqdm import tqdm
from latest_utils import SimilarityComputer


# ============= 评估函数 =============
def i2t(npts, sims, return_ranks=False):
    """
    计算图像到文本的检索精度
    Args:
        npts: 图像数量
        sims: 相似度矩阵 (N, 5N)
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]  # 按相似度降序排列

        # 找到对应5个文本的位置
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # 计算指标
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(npts, sims, return_ranks=False):
    """
    计算文本到图像的检索精度
    Args:
        npts: 图像数量
        sims: 相似度矩阵 (N, 5N)
    """
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    sims = sims.T  # 转置矩阵 (5N, N)

    for index in range(npts):
        for i in range(5):
            # 当前文本索引
            text_index = 5 * index + i
            inds = np.argsort(sims[text_index])[::-1]  # 按相似度降序排列
            ranks[text_index] = np.where(inds == index)[0][0]
            top1[text_index] = inds[0]

    # 计算指标
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def validate(model, val_loader, device):
    """在验证集上评估模型性能（完整版：包含局部特征）"""
    model.eval()
    # all_img_ids = []
    # all_caption_ids = []
    embed_dim = model.embed_dim
    shard_size = 100  # 根据GPU显存调整分块大小

    # 初始化特征存储
    n_data = len(val_loader.dataset)

    # cap_lens = []
    attention_masks = []
    img_embs = []
    local_img_embs = []
    cap_embs = []
    local_cap_embs = []

    # 收集所有图像和文本特征
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc="Extracting Features")):
            images = batch['images'].to(device)
            texts = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            # img_ids = batch['image_id']
            # caption_ids = batch['caption_ids']

            # 获取特征
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images, texts, attn_mask)

            # 确保模型返回了局部特征
            assert "global_vis" in outputs and "local_vis" in outputs and "text_feats" in outputs and "local_text" in outputs, \
                "模型必须返回全局和局部特征"

            global_vis = outputs["global_vis"]
            local_vis = outputs["local_vis"]
            text_feats = outputs["text_feats"]
            local_text = outputs["local_text"]

            attention_masks.append(attn_mask)
            img_embs.append(global_vis)
            local_img_embs.append(local_vis)
            cap_embs.append(text_feats)
            local_cap_embs.append(local_text)

        # 合并注意力掩码
        attention_masks = torch.cat(attention_masks, dim=0)
        attention_masks = attention_masks.view(-1, attention_masks.size(-1))
        img_embs = torch.cat(img_embs, dim=0)
        local_img_embs = torch.cat(local_img_embs, dim=0)
        cap_embs = torch.cat(cap_embs, dim=0)
        cap_embs = cap_embs.view(-1, embed_dim)
        local_cap_embs = torch.cat(local_cap_embs, dim=0)
        local_cap_embs = local_cap_embs.view(-1, local_cap_embs.size(-2) ,embed_dim)

        # 计算相似度矩阵
        sims = shard_attn_scores(model, img_embs, local_img_embs, cap_embs, local_cap_embs, attention_masks, device)

        # 计算评估指标
        n_img = len(img_embs)
        i2t_results = i2t(n_img, sims)
        t2i_results = t2i(n_img, sims)

        r1_i2t, r5_i2t, r10_i2t, medr_i2t, meanr_i2t = i2t_results
        r1_t2i, r5_t2i, r10_t2i, medr_t2i, meanr_t2i = t2i_results
        rsum = r1_i2t + r5_i2t + r10_i2t + r1_t2i + r5_t2i + r10_t2i

        print(f"\nValidation Results:")
        print(f"Image to Text: R@1={r1_i2t:.1f}, R@5={r5_i2t:.1f}, R@10={r10_i2t:.1f}")
        print(f"Text to Image: R@1={r1_t2i:.1f}, R@5={r5_t2i:.1f}, R@10={r10_t2i:.1f}")
        print(f"RSUM: {rsum:.1f}")

    return (r1_i2t, r5_i2t, r10_i2t), (r1_t2i, r5_t2i, r10_t2i), rsum


def shard_attn_scores(model, img_embs, local_img_embs, cap_embs, local_cap_embs, attention_masks, device,
                      shard_size=20):
    """
    分块计算相似度矩阵（包含局部特征）
    """
    n_im_shard = (img_embs.size(0) - 1) // shard_size + 1
    n_cap_shard = (cap_embs.size(0) - 1) // shard_size + 1
    sims = np.zeros((len(img_embs), len(cap_embs)))
    # sims = torch.zeros(img_embs.size(0), cap_embs.size(0)).to(device)

    for i in range(n_im_shard):
        im_start = i * shard_size
        im_end = min((i + 1) * shard_size, img_embs.size(0))

        for j in range(n_cap_shard):
            cap_start = j * shard_size
            cap_end = min((j + 1) * shard_size, cap_embs.size(0))

            # 获取当前分块数据
            im_batch = img_embs[im_start:im_end]
            local_im_batch = local_img_embs[im_start:im_end]
            cap_batch = cap_embs[cap_start:cap_end]
            local_cap_batch = local_cap_embs[cap_start:cap_end]
            attn_batch = attention_masks[cap_start:cap_end]

            # 确保维度正确
            # if len(im_batch.size()) == 1:
            #     im_batch = im_batch.unsqueeze(0)
            # if len(cap_batch.size()) == 1:
            #     cap_batch = cap_batch.unsqueeze(0)
            # if len(local_im_batch.size()) == 2:
            #     local_im_batch = local_im_batch.unsqueeze(0)
            # if len(local_cap_batch.size()) == 2:
            #     local_cap_batch = local_cap_batch.unsqueeze(0)

            # 计算相似度（传递所有必要特征）
            with torch.no_grad():
                sim_batch = model.similarity_computer(
                    attn_batch,
                    im_batch,
                    cap_batch,
                    local_im_batch,
                    local_cap_batch,
                )

            sims[im_start:im_end, cap_start:cap_end] = sim_batch.data.cpu().numpy()
            del im_batch, local_im_batch, cap_batch, local_cap_batch, attn_batch
            torch.cuda.empty_cache()

    return sims


# def validate(model, val_loader, device):
#     """在验证集上评估模型性能"""
#     model.eval()
#     all_img_ids = []
#     all_caption_ids = []
#     all_sim_matrices = []
#     embed_dim = model.embed_dim
#     model.similarity_computer = SimilarityComputer(embed_dim, mode="validate")
#
#     with torch.no_grad():
#         for batch in tqdm(val_loader, desc="Validating"):
#             images = batch['images'].to(device)
#             batch_size = images.size(0)
#             texts = batch['input_ids'].view(batch_size, 5, 77).to(device)
#             attention_mask = batch['attention_mask'].view(batch_size, 5, 77).to(device)
#             img_ids = batch['image_id']
#             caption_ids = batch['caption_ids']
#
#             # 前向传播
#             outputs = model(images, texts, attention_mask)
#             sim_matrix = outputs['sim_matrix'].cpu().numpy()  # [B, 5B] 或 [B, B*5]
#
#             # 存储结果
#             all_img_ids.extend(img_ids)
#             all_caption_ids.extend(caption_ids)
#             all_sim_matrices.append(sim_matrix)
#
#         # 合并所有批次的相似度矩阵
#         full_sim_matrix = np.concatenate(all_sim_matrices, axis=0)
#         n_image = len(all_img_ids)
#
#         # 确保文本顺序正确 (image1_cap1, image1_cap2, ..., image2_cap1, ...)
#         sorted_indices = []
#         for i, img_id in enumerate(all_img_ids):
#             for j in range(5):  # 每个图像5个文本
#                 text_idx = i * 5 + j
#                 sorted_indices.append(text_idx)
#
#         # 重新排列相似度矩阵
#         sorted_sim_matrix = full_sim_matrix[:, sorted_indices]
#
#         # 计算评估指标
#         i2t_results = i2t(n_image, sorted_sim_matrix)
#         t2i_results = t2i(n_image, sorted_sim_matrix)
#
#         # 解析结果
#         r1_i2t, r5_i2t, r10_i2t, medr_i2t, meanr_i2t = i2t_results
#         r1_t2i, r5_t2i, r10_t2i, medr_t2i, meanr_t2i = t2i_results
#
#         # 计算rsum
#         rsum = r1_i2t + r5_i2t + r10_i2t + r1_t2i + r5_t2i + r10_t2i
#
#         print(f"\nValidation Results:")
#         print(f"Image to Text: R@1={r1_i2t:.1f}, R@5={r5_i2t:.1f}, R@10={r10_i2t:.1f}")
#         print(f"Text to Image: R@1={r1_t2i:.1f}, R@5={r5_t2i:.1f}, R@10={r10_t2i:.1f}")
#         print(f"RSUM: {rsum:.1f}")
#
#     return (r1_i2t, r5_i2t, r10_i2t), (r1_t2i, r5_t2i, r10_t2i), rsum


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import pywt
from transformers import BertModel, BertConfig
from timm.models.vision_transformer import Block


class AdvancedCrossModalRetriever(nn.Module):
    """高级图文检索网络：优化特征增强路径"""

    def __init__(self, clip_model, num_crops=5, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_crops = num_crops

        # === 全局特征路径：傅里叶增强 ===
        self.global_path = GlobalFourierPath(clip_model, embed_dim)

        # === 局部特征路径：小波增强 ===
        self.local_path = LocalWaveletPath(clip_model, embed_dim, num_crops)

        # === 上下文场景增强模块 ===
        self.context_enhancer = ContextEnhancer(embed_dim)

        # === 文本编码器 ===
        self.text_encoder = TextEncoder(embed_dim)

        # === 图相关推理模块 ===
        self.visual_graph = HierarchicalGraphEncoder(embed_dim)
        self.text_graph = HierarchicalGraphEncoder(embed_dim)
        self.cross_graph = CrossGraphReasoner(embed_dim)

        # === 特征净化模块 ===
        self.feature_purifier = CrossModalPurifier(embed_dim)

        # === 多匹配关系建模 ===
        self.multi_match = MultiMatchModule(embed_dim)

        # === 损失模块 ===
        self.loss_module = MultiLevelLoss(embed_dim)

    def forward(self, images, texts, attention_mask=None):
        # ===== 特征提取 =====
        # 全局特征 (傅里叶增强)
        global_feat = self.global_path(images[:, 0])

        # 局部特征 (小波增强)
        local_feats = self.local_path(images[:, 1:1 + self.num_crops])

        # 场景上下文增强
        context_feat = self.context_enhancer(global_feat, local_feats)

        # 文本特征
        text_feats = self.text_encoder(texts, attention_mask)

        # ===== 特征净化 =====
        puri_vis, puri_text = self.feature_purifier(local_feats, text_feats, context_feat)

        # ===== 图相关推理 =====
        vis_graph = self.visual_graph(puri_vis)
        text_graph = self.text_graph(puri_text)
        fused_graph = self.cross_graph(vis_graph, text_graph)

        # ===== 多匹配关系建模 =====
        match_scores = self.multi_match(context_feat, local_feats, text_feats, fused_graph)

        return {
            "global_feat": context_feat,
            "local_feats": local_feats,
            "text_feats": text_feats,
            "fused_graph": fused_graph,
            "match_scores": match_scores
        }

    def compute_loss(self, outputs, labels):
        return self.loss_module(outputs, labels)


# ===== 核心模块实现 =====

class GlobalFourierPath(nn.Module):
    """全局傅里叶路径：增强结构信息"""

    def __init__(self, clip_model, embed_dim):
        super().__init__()
        self.clip = clip_model
        self.embed_dim = embed_dim

        # 傅里叶变换
        self.fourier = FourierEnhancement()

        # 低频增强模块
        self.low_freq_enhancer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # CLIP特征增强
        self.clip_enhancer = nn.Sequential(
            nn.Linear(512, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, global_img):
        # 原始CLIP特征
        with torch.no_grad():
            orig_clip_feat = self.clip.encode_image(global_img)

        # 傅里叶增强
        fourier_img = self.fourier(global_img)

        # 低频特征提取
        low_freq_feat = self.low_freq_enhancer(fourier_img).flatten(1)

        # 傅里叶增强的CLIP特征
        with torch.no_grad():
            fourier_clip_feat = self.clip.encode_image(fourier_img)

        # 特征融合
        fused_feat = torch.cat([
            self.clip_enhancer(orig_clip_feat.float()),
            self.clip_enhancer(fourier_clip_feat.float()),
            low_freq_feat
        ], dim=1)

        return nn.Linear(fused_feat.size(1), self.embed_dim)(fused_feat)


class LocalWaveletPath(nn.Module):
    """局部小波路径：增强细节信息"""

    def __init__(self, clip_model, embed_dim, num_crops):
        super().__init__()
        self.clip = clip_model
        self.embed_dim = embed_dim
        self.num_crops = num_crops

        # 小波增强
        self.wavelet = WaveletEnhancement()

        # 高频增强模块
        self.high_freq_enhancer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((7, 7))
        )

    def forward(self, crops):
        """crops: [B, num_crops, 3, 224, 224]"""
        batch_size = crops.size(0)
        enhanced_feats = []

        for i in range(self.num_crops):
            crop = crops[:, i]

            # 小波增强
            wave_crop = self.wavelet(crop)

            # 原始CLIP特征
            with torch.no_grad():
                orig_feat = self.clip.encode_image(crop)

            # 小波增强的CLIP特征
            with torch.no_grad():
                wave_feat = self.clip.encode_image(wave_crop)

            # 高频特征提取
            high_freq_feat = self.high_freq_enhancer(wave_crop).flatten(1)

            # 特征融合
            fused_feat = torch.cat([
                orig_feat.float(),
                wave_feat.float(),
                high_freq_feat
            ], dim=1)

            enhanced_feats.append(nn.Linear(fused_feat.size(1), self.embed_dim)(fused_feat))

        return torch.stack(enhanced_feats, dim=1)


class ContextEnhancer(nn.Module):
    """场景上下文增强模块"""

    def __init__(self, embed_dim):
        super().__init__()
        # 场景理解Transformer
        self.context_transformer = nn.Sequential(
            Block(embed_dim, num_heads=8),
            Block(embed_dim, num_heads=8)
        )

        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(embed_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, global_feat, local_feats):
        # 将全局特征转换为空间形式
        B, D = global_feat.shape
        global_spatial = global_feat.view(B, D, 1, 1).expand(-1, -1, 7, 7)

        # 将局部特征聚合为空间图
        local_aggregated = torch.mean(local_feats, dim=1).view(B, D, 1, 1).expand(-1, -1, 7, 7)

        # 场景特征融合
        scene_feats = torch.cat([global_spatial, local_aggregated], dim=1)

        # 上下文理解
        scene_feats = scene_feats.flatten(2).permute(0, 2, 1)  # [B, 49, 2D]
        scene_feats = self.context_transformer(scene_feats)
        scene_feats = scene_feats.permute(0, 2, 1).view(B, -1, 7, 7)

        # 空间注意力加权
        attn_weights = self.spatial_attn(scene_feats)
        enhanced_feats = attn_weights * scene_feats

        # 全局池化
        return torch.mean(enhanced_feats, dim=[2, 3])


class TextEncoder(nn.Module):
    """文本编码器：BERT+语义增强"""

    def __init__(self, embed_dim):
        super().__init__()
        self.bert = BertModel(BertConfig(
            hidden_size=embed_dim,
            num_hidden_layers=4,
            num_attention_heads=8
        ))

        # 语义增强
        self.semantic_enhancer = nn.Sequential(
            Block(embed_dim, num_heads=8),
            Block(embed_dim, num_heads=8)
        )

    def forward(self, texts, attention_mask):
        outputs = self.bert(input_ids=texts, attention_mask=attention_mask)
        text_feats = outputs.last_hidden_state
        return self.semantic_enhancer(text_feats)


class HierarchicalGraphEncoder(nn.Module):
    """层级图编码器：节点级+图级"""

    def __init__(self, embed_dim):
        super().__init__()
        # 节点级编码
        self.node_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # 图注意力网络
        self.gat = GraphAttentionNetwork(embed_dim)

        # 图级聚合
        self.graph_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, features):
        # 节点编码
        node_feats = self.node_encoder(features)

        # 图注意力
        attn_feats = self.gat(node_feats)

        # 图级特征
        graph_feat = self.graph_pool(attn_feats.permute(0, 2, 1)).squeeze(-1)
        return {"node": attn_feats, "graph": graph_feat}


class GraphAttentionNetwork(nn.Module):
    """图注意力网络：GATv2实现"""

    def __init__(self, embed_dim, heads=4):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_dim // heads

        # 线性变换
        self.W = nn.Linear(embed_dim, heads * self.head_dim)
        self.a = nn.Parameter(torch.empty(1, heads, 2 * self.head_dim))

        # 初始化
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x):
        B, N, _ = x.shape

        # 线性变换
        h = self.W(x).view(B, N, self.heads, self.head_dim)

        # 计算注意力系数
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1, -1)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1, -1)
        a_input = torch.cat([h_i, h_j], dim=-1)

        e = torch.einsum('bijkh,nh->bijn', a_input, self.a.squeeze(0))
        e = F.leaky_relu(e, 0.2)

        # 注意力权重
        attention = F.softmax(e, dim=-1)

        # 加权聚合
        output = torch.einsum('bijn,bjnh->binh', attention, h)
        return output.reshape(B, N, -1)


class CrossGraphReasoner(nn.Module):
    """跨图推理器：双向图注意力"""

    def __init__(self, embed_dim):
        super().__init__()
        # 视觉到文本的图注意力
        self.vis_to_text = GraphAttentionNetwork(embed_dim)

        # 文本到视觉的图注意力
        self.text_to_vis = GraphAttentionNetwork(embed_dim)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, vis_graph, text_graph):
        # 节点特征
        vis_nodes = vis_graph["node"]
        text_nodes = text_graph["node"]

        # 视觉→文本
        vis_to_text = self.vis_to_text(
            torch.cat([vis_nodes, text_nodes], dim=1)
        )[:, :vis_nodes.size(1)]

        # 文本→视觉
        text_to_vis = self.text_to_vis(
            torch.cat([text_nodes, vis_nodes], dim=1)
        )[:, :text_nodes.size(1)]

        # 特征融合
        fused_feats = torch.cat([vis_to_text, text_to_vis], dim=-1)
        return self.fusion(fused_feats)


class CrossModalPurifier(nn.Module):
    """跨模态特征净化器：门控注意力机制"""

    def __init__(self, embed_dim):
        super().__init__()
        # 视觉净化门
        self.vis_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.vis_purifier = nn.Linear(embed_dim, embed_dim)

        # 文本净化门
        self.text_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.text_purifier = nn.Linear(embed_dim, embed_dim)

    def forward(self, vis_feats, text_feats, global_vis):
        # 文本引导的视觉净化
        text_guide = torch.mean(text_feats, dim=1, keepdim=True)
        vis_gate = self.vis_gate(torch.cat([vis_feats, text_guide.expand_as(vis_feats)], dim=-1))
        puri_vis = self.vis_purifier(vis_gate * vis_feats)

        # 视觉引导的文本净化
        vis_guide = global_vis.unsqueeze(1)
        text_gate = self.text_gate(torch.cat([text_feats, vis_guide.expand_as(text_feats)], dim=-1))
        puri_text = self.text_purifier(text_gate * text_feats)

        return puri_vis, puri_text


class MultiMatchModule(nn.Module):
    """多匹配关系建模：三路相似度"""

    def __init__(self, embed_dim):
        super().__init__()
        # 全局匹配
        self.global_match = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        # 局部匹配
        self.local_match = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        # 图增强匹配
        self.graph_match = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, global_vis, local_vis, text_feats, fused_graph):
        batch_size, num_crops, _ = local_vis.shape
        num_texts = text_feats.size(1)

        # 全局匹配分数
        global_text = torch.mean(text_feats, dim=1)
        global_score = self.global_match(torch.cat([global_vis, global_text], dim=1))

        # 局部匹配分数
        local_scores = []
        for i in range(num_crops):
            for j in range(num_texts):
                feat_pair = torch.cat([local_vis[:, i], text_feats[:, j]], dim=1)
                score = self.local_match(feat_pair)
                local_scores.append(score)
        local_scores = torch.stack(local_scores, dim=1).view(batch_size, num_crops, num_texts)

        # 图增强匹配
        graph_scores = []
        for i in range(num_crops):
            for j in range(num_texts):
                feat_triplet = torch.cat([
                    local_vis[:, i],
                    text_feats[:, j],
                    fused_graph[:, i]
                ], dim=1)
                score = self.graph_match(feat_triplet)
                graph_scores.append(score)
        graph_scores = torch.stack(graph_scores, dim=1).view(batch_size, num_crops, num_texts)

        # 组合所有分数
        return torch.cat([
            global_score.unsqueeze(1).unsqueeze(1).expand(-1, num_crops, num_texts),
            local_scores,
            graph_scores
        ], dim=-1)  # [B, num_crops, num_texts, 3]


class MultiLevelLoss(nn.Module):
    """多层级损失：结合CoCa和对比学习"""

    def __init__(self, embed_dim):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, outputs, labels):
        # 提取特征
        global_vis = outputs["global_feat"]
        text_feats = outputs["text_feats"]
        match_scores = outputs["match_scores"]

        # === 实例级对比损失 (CoCa风格) ===
        global_text = torch.mean(text_feats, dim=1)
        logits = torch.matmul(global_vis, global_text.t()) / self.temperature
        inst_loss = F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)

        # === 局部对比损失 ===
        local_vis = outputs["local_feats"]
        local_loss = 0
        for i in range(local_vis.size(1)):
            crop_feats = local_vis[:, i]
            logits = torch.matmul(crop_feats, text_feats.mean(dim=1).t()) / self.temperature
            local_loss += F.cross_entropy(logits, labels)
        local_loss /= local_vis.size(1)

        # === 图一致性损失 ===
        vis_graph = outputs["fused_graph"]
        text_graph = outputs["text_feats"]
        graph_loss = F.mse_loss(
            F.normalize(vis_graph.mean(dim=1), dim=-1),
            F.normalize(text_graph.mean(dim=1), dim=-1)
        )

        # === 多匹配损失 ===
        match_loss = self.matching_loss(match_scores, labels)

        # 总损失
        total_loss = 0.4 * inst_loss + 0.3 * local_loss + 0.2 * graph_loss + 0.1 * match_loss
        return total_loss

    def matching_loss(self, match_scores, labels):
        """多匹配铰链损失"""
        batch_size, num_crops, num_texts, _ = match_scores.shape
        loss = 0

        for i in range(batch_size):
            # 正样本分数 (匹配对)
            pos_scores = match_scores[i, :, i * 5:(i + 1) * 5, :].mean(dim=(1, 2))

            # 负样本分数 (不匹配对)
            neg_indices = torch.arange(batch_size) != i
            neg_scores = match_scores[i, :, neg_indices].mean(dim=(1, 2))

            # 铰链损失
            max_neg = torch.max(neg_scores)
            loss += F.relu(0.2 + max_neg - torch.mean(pos_scores))

        return loss / batch_size


# ===== 特征增强模块 =====

class FourierEnhancement(nn.Module):
    """傅里叶增强：聚焦低频结构信息"""

    def forward(self, x):
        # 傅里叶变换
        f = torch.fft.fft2(x)
        fshift = torch.fft.fftshift(f)

        # 创建低通滤波器
        _, _, h, w = x.shape
        mask = torch.zeros(h, w).to(x.device)
        cx, cy = h // 2, w // 2
        mask[cx - 30:cx + 30, cy - 30:cy + 30] = 1.0

        # 应用低通滤波
        magnitude = torch.abs(fshift) * mask
        phase = torch.angle(fshift)

        # 重构图像
        f_ishift = torch.fft.ifftshift(magnitude * torch.exp(1j * phase))
        img_recon = torch.fft.ifft2(f_ishift)
        return torch.abs(img_recon)


class WaveletEnhancement(nn.Module):
    """小波增强：聚焦高频细节信息"""

    def __init__(self, wavelet='bior4.4'):
        super().__init__()
        self.wavelet = wavelet

    def forward(self, x):
        # 小波变换
        coeffs = pywt.dwt2(x.cpu().permute(0, 2, 3, 1).numpy(), self.wavelet)
        cA, (cH, cV, cD) = coeffs

        # 增强高频细节
        cH = np.clip(cH * 2.0, -1, 1)
        cV = np.clip(cV * 1.8, -1, 1)
        cD = np.clip(cD * 1.5, -1, 1)

        # 重构图像
        reconstructed = pywt.idwt2((cA, (cH, cV, cD)), self.wavelet)
        return torch.tensor(reconstructed).permute(0, 3, 1, 2).to(x.device)


# ===== 训练主函数 =====

def train_retriever():
    import clip
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader
    import torch.optim as optim

    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载CLIP模型
    clip_model, _ = clip.load("ViT-B/32", device=device)
    for param in clip_model.parameters():
        param.requires_grad = False

    # 初始化数据集
    annotation_path = "flickr30k/results.csv"
    image_dir = "flickr30k/images"
    rpn_file = "flickr30k/rpn_proposals.json"
    annotation_df = load_flickr_annotations(annotation_path)
    dataset = Flickr30kDataset(image_dir, annotation_df, rpn_proposals_file=rpn_file)

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # 初始化模型
    model = AdvancedCrossModalRetriever(clip_model).to(device)

    # 优化器配置
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=len(dataloader) * 5,
        T_mult=1,
        eta_min=1e-6
    )

    # 训练循环
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, (images, texts) in enumerate(dataloader):
            if images is None:
                continue

            images = images.to(device)
            texts = texts.to(device)

            # 创建标签 (batch_size)
            batch_size = images.size(0)
            labels = torch.arange(batch_size).to(device)

            optimizer.zero_grad()

            # 前向传播
            outputs = model(images, texts)

            # 计算损失
            loss = model.compute_loss(outputs, labels)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")

        # 保存模型
        torch.save(model.state_dict(), f"advanced_retriever_epoch{epoch}.pth")

        # 验证逻辑
        # ...


if __name__ == "__main__":
    train_retriever()
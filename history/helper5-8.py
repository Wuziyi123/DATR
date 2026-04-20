import gc
import os
import random

import cv2
import numpy as np
import torch
import json
import pickle

from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from clip import clip
from torchvision.datasets import ImageNet, ImageFolder, Places365
from my_datasets import *
from utils import (
    openai_imagenet_classes,
    imagenet_classes,
    imagenet_a_lt,
    imagenet_r_lt,
)
from sam_sample import get_crop_Images
from edge_sam import SamPredictor, sam_model_registry
from skimage.transform import resize


def load_json(filename):
    if not filename.endswith(".json"):
        filename += ".json"
    with open(filename, "r") as fp:
        return json.load(fp)


def set_seed(seed):
    print(f"Setting seed {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_dataset(data_path, dataset_name, custom_loader):
    data_path = data_path

    if dataset_name == MyDataset.ImageNet:
        dataset = ImageNet(
            data_path,
            split="val",
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.ImageNetV2:
        dataset = ImageNetV2Dataset(
            location=data_path,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.ImageNetR:
        dataset = ImageFolder(
            root=data_path,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.ImageNetS:
        dataset = ImageFolder(
            root=data_path,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.ImageNetA:
        dataset = ImageFolder(
            root=data_path,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.CUB:
        dataset = CUBDataset(
            data_path,
            train=False,
            transform=None,
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.Food101:
        dataset = Food101(
            data_path,
            transform=None,
            loader=custom_loader,
            split="test",
            download=False,
        )

    elif dataset_name == MyDataset.OxfordIIITPet:
        dataset = OxfordIIITPet(
            data_path,
            transform=None,
            split="test",
            loader=custom_loader,
        )

    elif dataset_name == MyDataset.Place365:
        dataset = Places365(
            data_path,
            transform=None,
            loader=custom_loader,
            download=False,
            split="val",
            small=False,
        )

    elif dataset_name == MyDataset.DTD:
        dataset = DTD(
            data_path,
            # transform=None,
            loader=custom_loader,
            split="test",
            download=False,
        )

    return dataset


def wordify(string):
    word = string.replace("_", " ")
    return word


def load_classes(dataset_name):
    with open(
        f"features/{dataset_name}/{dataset_name}.json",
        "r",
    ) as f:
        classes = json.load(f)

    wordify_classes = []
    for c in classes:
        wordify_classes.append(wordify(c))

    return wordify_classes


def generate_weights(
    method,
    model,
    dataset_name,
    tt_scale=None,
    device=None,
):
    templates = None
    make_sentence = False
    is_template = True

    # if dataset start with imagenet
    if dataset_name.startswith(MyDataset.ImageNet):
        classes = (
            openai_imagenet_classes
            if method in ["clip-d", "waffle"]
            else imagenet_classes
        )
    else:
        classes = load_classes(dataset_name)

    print(f"Creating {method} text embeddings...")

    if method != "clip":
        if method == "ours":
            load_file = "cupl"
        elif method == "cupl":
            load_file = "cupl"
        elif method == "waffle":
            load_file = "clip-d"
        else:
            load_file = method

        with open(f"prompts/{dataset_name}/{load_file}.json") as f:
            templates = json.load(f)

        if method in ["waffle", "clip-d", "cupl", "ours"]:
            is_template = False

        if method == "clip-d":
            make_sentence = True

        if method == "waffle":
            templates = construct_random(templates)

    zeroshot_weights = zeroshot_classifier(
        model,
        classes,
        templates,
        is_template,
        make_sentence,
        tt_scale,
        device,
    )

    return zeroshot_weights


class SAMEnhancedCLIP(torch.nn.Module):
    def __init__(self, clip_model, device):
        super().__init__()
        self.device = device
        self.patch_size = 16
        sam = sam_model_registry["edge_sam"](checkpoint="edge_sam_3x.pth")
        sam.half().to(device).eval()
        self.predictor = SamPredictor(sam)
        self.clip = clip_model
        self.num_heads = 12
        # 注册hook处理中间特征
        self.attention_masks = None
        # self.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, outputs):
        """在每层Transformer前注入掩码"""
        if self.attention_masks is not None:
            # for layer in self.vit.encoder.layers:
            #     layer.self_attn.attention_mask = self.attention_masks
            for layer in self.clip.visual.transformer.resblocks:
                layer.attn_mask = self.attention_masks

    def _register_attention_hooks(self):
        """在每层Transformer前注入掩码"""
        if self.attention_masks is not None:
            for layer in self.clip.visual.transformer.resblocks:
                layer.attn_mask = self.attention_masks
        else:
            for layer in self.clip.visual.transformer.resblocks:
                layer.attn_mask = None

    def _prepare_mask(self, binary_mask):
        """将原始二值掩码转换为注意力引导矩阵"""
        # Step 1: 下采样到patch网格
        # grid_size = 224 // self.patch_size
        mask_patches = F.avg_pool2d(
            binary_mask.float(),
            kernel_size=self.patch_size,
            stride=self.patch_size
        ).squeeze()  # [14,14]

        # Step 2: 构建序列掩码
        batch_size = mask_patches.size(0)
        cls_tokens = torch.ones(batch_size, 1, device=self.device)  # 批量CLS token
        seq_mask = torch.cat([
            cls_tokens,
            mask_patches.flatten(start_dim=1)
        ], dim=1)  # [197]

        # Step 3: 生成注意力引导矩阵 (优化广播机制)
        attn_guidance = torch.einsum(
            'bi,bj->bij',
            seq_mask,
            seq_mask
        ) * 3.0  # 外积运算 [B, 197, 197]

        # Step 4: 扩展到多头 [B, num_heads, 197, 197]
        attn_guidance = attn_guidance.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        # 合并批次和头维度
        attn_mask_3d = attn_guidance.reshape(-1, attn_guidance.size(2), attn_guidance.size(3))
        return attn_mask_3d

    def forward(self, x, sampling_points, paths):
        # 重置注意力为空
        self.attention_masks = None
        self._register_attention_hooks()
        # 生成掩码 304 * 512 == 16 * (1+18) * 512
        with torch.no_grad():
            B, _, C, H, W = x.shape  # 16 19 3 224 224
            batch_masks = []
            image_global = x[:, 0, :, :, :].half()
            image_crop = x[:, 1:, :, :, :].reshape(-1, C, H, W).contiguous().half()
            outputs_crop = self.clip.encode_image(image_crop)

            for i in range(len(paths)):
                image = cv2.imread(paths[i])
                image_plt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.predictor.set_image(image_plt)

                with torch.cuda.amp.autocast():
                    masks, scores, logits = self.predictor.predict(
                        point_coords=np.array(sampling_points[i]),
                        point_labels=np.ones(len(sampling_points[i]), dtype=np.int64),
                        num_multimask_outputs=1,
                        use_stability_score=True
                    )
                mask_tensor = torch.from_numpy(masks[0].astype(np.float32))
                mask_4d = mask_tensor.unsqueeze(0).unsqueeze(0)
                resized = F.interpolate(mask_4d, size=(224, 224),mode='nearest').squeeze().numpy()  # 移除通道维度 → [B,H,W]

                # 确保严格二值化
                binary_mask = (resized >= 0.5).astype(np.float16)
                batch_masks.append(binary_mask)

            tensors = [torch.from_numpy(arr) for arr in batch_masks]
            mask_imgs = torch.stack(tensors).to(self.device)

            # 生成注意力引导矩阵
            self.attention_masks = self._prepare_mask(mask_imgs)
            # 原始ViT处理流程
            self._register_attention_hooks()
            outputs_global = self.clip.encode_image(image_global)

            # 计算插入后的总长度
            total_length = outputs_global.shape[0] + outputs_crop.shape[0]
            result = torch.zeros(total_length, outputs_global.shape[1], dtype=outputs_crop.dtype, device=self.device)  # (304, 512)
            index = total_length // outputs_global.shape[0]
            # 生成插入位置的索引（每隔18个位置插入1个）
            insert_positions = torch.arange(0, total_length, index, device=self.device)  # [0, 19, 38, 57, ..., 285]
            # 生成非插入位置的索引（即 outputs_crop 应该填充的位置）
            non_insert_mask = torch.ones(total_length, dtype=torch.bool, device=self.device)  # 初始全 True
            non_insert_mask[insert_positions] = False  # 插入位置设为 False
            non_insert_indices = torch.where(non_insert_mask)[0]  # 获取非插入位置的索引

            # 填充 outputs_crop 到非插入位置
            result[non_insert_indices] = outputs_crop
            # 填充 outputs_global 到插入位置
            result[insert_positions] = outputs_global

        return result


def load_precomputed_features(
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
    save_file = (dataset_name + "-" + model_size).replace("/", "-")
    save_root = f"features/{dataset_name}"

    # if save_root not exist, create it
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    filename = os.path.join(save_root, f"{save_file}-{alpha}-{n_samples}.pkl")

    if os.path.exists(filename):
        print(f"Loading {filename}...")
        load_res = pickle.load(open(filename, "rb"))
    else:
        print(f"File {filename} not found, precomputing features...")
        dataset = load_dataset(
            data_path=data_path,
            dataset_name=dataset_name,
            custom_loader=custom_loader,
        )

        dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # CUB classes need to be manually processed
        # if hasattr(dataset, "classes") and dataset_name != MyDataset.CUB:
        #     classes = dataset.classes
        #     classes_file = os.path.join(save_root, f"{dataset_name}.json")
        #     if not os.path.exists(classes_file):
        #         with open(classes_file, "w") as f:
        #             json.dump(classes, f)

        precomputed_features = []
        image_features_tensor = []
        target = []
        sam_enhanced_clip = SAMEnhancedCLIP(model, device)

        with torch.no_grad():
            for batch in tqdm(dataloader):
                # images = B=12, NS=1, C=3, H=224, W=224
                images, labels, paths = batch
                labels = labels.to(device)
                images = images.to(device, non_blocking=True)

                b, ns = images.shape[:2]
                # combined_images = combined_images.flatten(0, 1)
                # images = images.view(-1, *images.shape[2:])

                # 掩码增强
                batch_selected_points = get_crop_Images(paths)
                image_features = sam_enhanced_clip(images, batch_selected_points, paths)
                # image_features = model.encode_image(images)
                del images

                image_features = F.normalize(image_features)
                image_features = image_features.view(b, ns, -1).contiguous() # b,ns,d

                # 优化: 分离计算图并转移数据到CPU
                patch_features = image_features[:, 1:].detach()  # 分离计算图
                image_features_main = image_features[:, :1].detach()
                del image_features  # 及时释放

                weight_image = (image_features_main * patch_features).sum(
                    dim=-1, keepdim=True
                )
                # DS = (image_features * patch_features)  # 4,60,512 = (4,1,512) * （4,60,512）
                # weight_image = (batch, nsamples, 1)
                patch_with_weights = torch.cat([patch_features, weight_image], -1)

                # 优化: 数据存储到CPU
                precomputed_features.append(patch_with_weights.cpu())
                target.append(labels.cpu())
                image_features_tensor.append(image_features_main.squeeze(1).cpu())
                # 优化: 强制垃圾回收显存
                del patch_features, image_features_main, weight_image, patch_with_weights
                torch.cuda.empty_cache()

                # precomputed_features.append(patch_with_weights)
                # target.append(labels)
                # image_features_tensor.append(image_features.squeeze(1))

        # 最终合并时转换回原精度（如需）
        load_res = {
            "patches": torch.cat([x for x in precomputed_features], dim=0),
            "images": torch.cat(image_features_tensor, dim=0),
            "labels": torch.cat(target, dim=0),
        }
        # load_res = {
        #     "patches": torch.cat(precomputed_features, dim=0),
        #     "images": torch.cat(image_features_tensor, dim=0),
        #     "labels": torch.cat(target, dim=0),
        # }

        os.makedirs(save_root, exist_ok=True)
        pickle.dump(load_res, open(filename, "wb"))

    precomputed_features = load_res["patches"].to(device)
    target = load_res["labels"].to(device)
    image_features_tensor = load_res["images"].to(device)

    return precomputed_features, target, image_features_tensor


def make_descriptor_sentence(descriptor):
    if descriptor.startswith("a") or descriptor.startswith("an"):
        return f"which is {descriptor}"
    elif (
        descriptor.startswith("has")
        or descriptor.startswith("often")
        or descriptor.startswith("typically")
        or descriptor.startswith("may")
        or descriptor.startswith("can")
    ):
        return f"which {descriptor}"
    elif descriptor.startswith("used"):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"


def zeroshot_classifier(
    model,
    textnames,
    templates=None,
    is_template=True,
    make_sentence=False,
    tt_scale=None,
    device=None,
):
    with torch.no_grad():
        zeroshot_weights = []
        for i in tqdm(range(len(textnames))):
            if not is_template:
                texts = []
                for t in templates[textnames[i]]:
                    if make_sentence:
                        desc_sen = make_descriptor_sentence(t)
                        texts.append(f"{textnames[i]}, {desc_sen}")
                    else:
                        texts.append(t)
            elif templates:
                texts = [template.format(textnames[i]) for template in templates]
            else:
                texts = [f"a photo of a {textnames[i]}."]

            if i == 0:
                print(texts)

            if tt_scale is not None:
                label = f"a photo of a {textnames[i]}."
                label_tokens = clip.tokenize(label, truncate=True).to(device)
                label_embeddings = model.encode_text(label_tokens)
                label_embeddings /= label_embeddings.norm(dim=-1, keepdim=True)

            texts_tensor = clip.tokenize(texts, truncate=True).to(device)
            class_embeddings = model.encode_text(texts_tensor)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            if tt_scale is not None:  # (50,512) @ (512,1)
                weight = class_embeddings @ label_embeddings.T
                weight = (weight * tt_scale).softmax(dim=0)
                class_embedding = (class_embeddings * weight).sum(dim=0)
                class_embedding /= class_embedding.norm()
            else:
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights


def construct_random(gpt3_prompts):
    """
    reference: https://github.com/ExplainableML/WaffleCLIP.git
    """
    key_list = list(gpt3_prompts.keys())

    # Get complete list of available descriptions.
    descr_list = [list(values) for values in gpt3_prompts.values()]
    descr_list = np.array([x for y in descr_list for x in y])

    ### Descriptor Makers.
    structured_descriptor_builder = (
        lambda item, cls: f"A photo of a {wordify(cls)}, {make_descriptor_sentence(item)}."
    )

    word_list = pickle.load(open("features/word_list.pkl", "rb"))

    avg_num_words = int(
        np.max(
            [
                np.round(np.mean([len(wordify(x).split(" ")) for x in key_list])),
                1,
            ]
        )
    )
    avg_word_length = int(
        np.round(
            np.mean(
                [np.mean([len(y) for y in wordify(x).split(" ")]) for x in key_list]
            )
        )
    )
    word_list = [x[:avg_word_length] for x in word_list]

    # (Lazy solution) Extract list of available random characters from gpt description list. Ideally we utilize a separate list.
    character_list = [x.split(" ") for x in descr_list]
    character_list = [
        x.replace(",", "").replace(".", "")
        for x in np.unique([x for y in character_list for x in y])
    ]
    character_list = np.unique(list("".join(character_list)))

    num_spaces = (
        int(np.round(np.mean([np.sum(np.array(list(x)) == " ") for x in key_list]))) + 1
    )
    num_chars = int(
        np.ceil(np.mean([np.max([len(y) for y in x.split(" ")]) for x in key_list]))
    )

    num_chars += num_spaces - num_chars % num_spaces
    sample_key = ""

    for s in range(num_spaces):
        for _ in range(num_chars // num_spaces):
            sample_key += "a"
        if s < num_spaces - 1:
            sample_key += " "

    gpt3_prompts = {key: [] for key in gpt3_prompts.keys()}

    for key in key_list:
        for _ in range(15):
            base_word = ""
            for a in range(avg_num_words):
                base_word += np.random.choice(word_list, 1, replace=False)[0]
                if a < avg_num_words - 1:
                    base_word += " "
            gpt3_prompts[key].append(structured_descriptor_builder(base_word, key))
            noise_word = ""
            use_key = sample_key if len(key) >= len(sample_key) else key
            for c in sample_key:
                if c != " ":
                    noise_word += np.random.choice(character_list, 1, replace=False)[0]
                else:
                    noise_word += ", "
            gpt3_prompts[key].append(structured_descriptor_builder(noise_word, key))

    match_key = np.random.choice(key_list)
    gpt3_prompts = {key: gpt3_prompts[match_key] for key in key_list}
    for key in gpt3_prompts:
        gpt3_prompts[key] = [
            x.replace(wordify(match_key), wordify(key)) for x in gpt3_prompts[key]
        ]

    return gpt3_prompts


def accuracy(output, target, n, dataset_name):
    # Get index of the maximum value as prediction
    if dataset_name.startswith(MyDataset.ImageNetA):
        _, pred = output[:, imagenet_a_lt].max(1)
    elif dataset_name.startswith(MyDataset.ImageNetR):
        _, pred = output[:, imagenet_r_lt].max(1)
    else:
        _, pred = output.max(1)
    # Compare prediction with target
    correct = pred.eq(target)
    # Calculate top-1 accuracy
    return float(correct.float().sum().cpu().numpy()) / n * 100

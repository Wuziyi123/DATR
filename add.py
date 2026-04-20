import torch
import os
from huggingface_hub import hf_hub_download


def download_swin_weights():
    """下载Swin-Transformer-Base预训练权重"""
    # 方法1: 从HuggingFace下载
    try:
        from transformers import SwinModel
        model = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
        os.makedirs("./swin_weights", exist_ok=True)
        model.save_pretrained("./swin_weights")
        print("Swin权重下载完成")
    except:
        print("请手动下载权重或检查网络连接")

    # 方法2: 手动下载（备用）
    # 访问: https://huggingface.co/microsoft/swin-base-patch4-window7-224
    # 下载所有文件到 ./swin_weights 目录


# 执行下载
download_swin_weights()
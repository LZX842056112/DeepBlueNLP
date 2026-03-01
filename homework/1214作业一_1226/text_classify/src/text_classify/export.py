# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/26 20:25
Create User : 19410
Desc : 将Python训练好的模型持久化转换为通用的静态结构
"""
import json
import os

import torch

from .config import Config


def export_jit(cfg: Config, model_name: str = "best.pkl"):
    """
    将PyTorch训练好的模型转换为TorchScript格式
    :return:
    """
    # 1. 模型恢复
    tokenizer = cfg.tokenizer
    ckpt = torch.load(os.path.join(cfg.model_output_dir, model_name), map_location='cpu')
    net = ckpt['net']
    net.cpu().eval()

    # 2. 静态转换
    jit_net = torch.jit.trace(
        net,
        (torch.randint(0, 100, (10, 128)), torch.ones((10, 128)))
    )
    torch.jit.save(
        jit_net,
        os.path.join(cfg.model_output_dir, f"{os.path.splitext(model_name)[0]}.pt"),
        _extra_files={
            'label2ids.txt': json.dumps(tokenizer.label2ids, ensure_ascii=False),
            'token2ids.txt': json.dumps(tokenizer.token2ids, ensure_ascii=False)
        }
    )

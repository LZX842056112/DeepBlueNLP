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
    net.export = True
    net.cpu().eval()

    # 2. 静态转换
    jit_net = torch.jit.trace(
        net,
        (torch.randint(0, 100, (10, 128)), torch.ones((10, 128)))
    )
    _network_type = ''
    # noinspection PyBroadException
    try:
        _network_type = net.network_type
    except:
        pass
    torch.jit.save(
        jit_net,
        os.path.join(cfg.model_output_dir, f"{os.path.splitext(model_name)[0]}.pt"),
        _extra_files={
            'label2ids.txt': json.dumps(ckpt['label2ids'], ensure_ascii=False),
            'network_type.txt': _network_type
        }
    )


def export_onnx(cfg: Config, model_name: str = "best.pkl"):
    """
        将PyTorch训练好的模型转换为onnx格式
    :return:
    """
    # 1. 模型恢复
    tokenizer = cfg.tokenizer
    ckpt = torch.load(os.path.join(cfg.model_output_dir, model_name), map_location='cpu')
    net = ckpt['net']
    append_special_token = ckpt.get('append_special_token', False)
    max_length = ckpt.get('max_length', None)
    net.export = True
    net.cpu().eval()
    _network_type = ''
    # noinspection PyBroadException
    try:
        _network_type = net.network_type
    except:
        pass

    # 2. 转换为onnx格式
    onnx_file = os.path.join(cfg.model_output_dir, f"{os.path.splitext(model_name)[0]}.onnx")
    torch.onnx.export(
        net.cpu(),  # --dynamic only compatible with cpu
        (torch.randint(0, 100, (10, 128)), torch.ones((10, 128))),
        onnx_file,
        verbose=False,
        opset_version=14,
        do_constant_folding=True,
        input_names=['token_ids', 'token_masks'],
        output_names=['scores'],
        dynamic_axes={
            'token_ids': {0: 'bs', 1: 't'},
            'token_masks': {0: 'bs', 1: 't'},
            'scores': {0: 'bs', 1: 't'}
        }
    )

    # Checks
    import onnx  # pip install onnx==1.17.0

    model_onnx = onnx.load(onnx_file)
    onnx.checker.check_model(model_onnx)

    # 添加元数据
    d = {
        'label2ids.txt': json.dumps(ckpt['label2ids'], ensure_ascii=False),
        'network_type.txt': _network_type,
        'append_special_token': append_special_token,
        'max_length': max_length
    }
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, onnx_file)

# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/24 21:13
Create User : 19410
Desc : 配置文件对象，包含训练、推理相关的入参配置文件对象
"""
from dataclasses import dataclass

from .datas.tokenizer import Tokenizer


@dataclass
class Config:
    model_output_dir: str  # 模型输出文件夹路径
    summary_dir: str  # 日志输出路径
    tokenizer: Tokenizer  # 分词器

    train_file: str  # 训练数据对应文件
    eval_file: str  # 模型评估数据对应文件

    total_epoch: int
    batch_size: int  # 批次大小
    hidden_size: int  # 网络的隐层大小
    lr: float  # 模型训练学习率

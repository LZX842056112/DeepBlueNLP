# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/24 21:13
Create User : 19410
Desc : 配置文件对象，包含训练、推理相关的入参配置文件对象
"""
import os.path
from dataclasses import dataclass
from typing import Optional, Union, Dict

from .datas.tokenizer import Tokenizer


@dataclass
class Config:
    output_dir: Optional[str] = None  # 输出文件夹根目标
    tokenizer: Optional[Tokenizer] = None  # 分词器
    vocab_size: Optional[int] = None  # 词汇表大小
    label2id: Optional[Union[str, Dict[str, int]]] = None  # 类别标签名称到id的映射
    num_classes: Optional[int] = None  # 类别数目

    train_file: Optional[str] = None  # 训练数据对应文件
    eval_file: Optional[str] = None  # 模型评估数据对应文件

    total_epoch: Optional[int] = None
    batch_size: Optional[int] = None  # 批次大小
    lr: Optional[float] = None  # 模型训练学习率

    network_type: str = 'lstm'  # 网络类型 可选lstm、bert
    lstm_layers: int = 1  # LSTM的层数
    lstm_hidden_size: int = 768
    max_length: int = 512  # 最大输入文本长度限制，仅在部分模型结构中生效
    bert_path: Optional[str] = None  # Bert模型迁移路径
    freeze: Optional[Union[bool, int]] = None  # 给定迁移模型的时候冻结参数

    max_no_improved_epoch: int = 10  # 提前停止器参数 当连续n个epoch模型效果均没有提升的时候，提前结束模型训练

    device: str = "cpu"

    @property
    def model_output_dir(self) -> str:
        return os.path.join(self.output_dir, self.network_type, "models")

    @property
    def summary_dir(self) -> str:
        return os.path.join(self.output_dir, self.network_type, "logs")

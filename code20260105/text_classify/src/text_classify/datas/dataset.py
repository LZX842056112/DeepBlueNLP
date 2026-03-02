# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/24 21:23
Create User : 19410
Desc : 自定义文本分类相关的DataSet加载
"""
from typing import List

import torch
from torch.utils.data import Dataset

from .tokenizer import TokenizerBase


class TextClassifyDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str], tokenizer: TokenizerBase):
        super().__init__()

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        # 1. 获取index对应的原始文本和原始标签字符串
        text = self.texts[index]
        label = self.labels[index]

        # 2. 分词 + token2id转换值 + 标签id转换
        tokenizer_output = self.tokenizer(text=text, label=label)

        # 3. 结果返回
        return {
            'text': tokenizer_output.text,
            'tokens': tokenizer_output.tokens,
            'token_ids': torch.tensor(tokenizer_output.token_ids, dtype=torch.int64),
            'token_masks': torch.ones(len(tokenizer_output.token_ids), dtype=torch.float32),
            'label': tokenizer_output.label,
            'label_id': torch.tensor(tokenizer_output.label_id, dtype=torch.int64)
        }

    def __len__(self):
        return len(self.texts)

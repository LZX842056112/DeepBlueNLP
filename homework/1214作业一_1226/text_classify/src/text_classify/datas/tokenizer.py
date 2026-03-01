# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/24 21:21
Create User : 19410
Desc : 定义分词器相关操作
"""
from dataclasses import dataclass
from typing import List, Optional, Dict

from .utils import split_text_to_tokens


@dataclass
class TokenizerOutput:
    text: str
    tokens: List[str]
    token_ids: List[int]
    label: Optional[str] = None
    label_id: Optional[int] = 0


class Tokenizer:
    def __init__(self,
                 token2ids: Dict[str, int],  # token到id的映射字典
                 label2ids: Dict[str, int],  # 标签名称到id的映射字典
                 unk_token='<UNK>',
                 pad_token='<PAD>'
                 ):
        super().__init__()
        self.token2ids = token2ids
        self.unk_token_id = self.token2ids[unk_token]
        self.pad_token_id = self.token2ids[pad_token]
        self.label2ids = label2ids

    def __call__(self, text: str, label: Optional[str] = None) -> TokenizerOutput:
        # 1. 分词
        tokens = split_text_to_tokens(text)

        # 2. 将每个token转换为token id
        token_ids = [self.token2ids.get(token, self.unk_token_id) for token in tokens]

        # 3. 标签转换
        label_id = None
        if label is not None:
            label = str(label)
            label_id = self.label2ids[label]

        return TokenizerOutput(text, tokens, token_ids, label, label_id)

    @property
    def vocab_size(self):
        return len(self.token2ids)

    @property
    def num_classes(self):
        return len(self.label2ids)
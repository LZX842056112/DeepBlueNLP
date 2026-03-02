# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/24 21:21
Create User : 19410
Desc : 定义分词器相关操作
"""
from dataclasses import dataclass
from typing import List, Optional, Dict

from transformers import BertTokenizer

from .utils import split_text_to_tokens


@dataclass
class TokenizerOutput:
    text: str
    tokens: List[str]
    token_ids: List[int]
    label: Optional[str] = None
    label_id: Optional[int] = 0


class TokenizerBase:
    def __call__(self, text: str, label: Optional[str] = None):
        raise NotImplementedError("子类实现")

    @property
    def vocab_size(self):
        raise NotImplementedError("子类实现")

    @property
    def num_classes(self):
        raise NotImplementedError("子类实现")

    @property
    def pad_token_id(self):
        raise NotImplementedError("子类实现")

    @property
    def unk_token_id(self):
        raise NotImplementedError("子类实现")

    @property
    def token2ids(self):
        raise NotImplementedError("子类实现")

    @property
    def label2ids(self):
        raise NotImplementedError("子类实现")


class Tokenizer(TokenizerBase):
    def __init__(self,
                 token2ids: Dict[str, int],  # token到id的映射字典
                 label2ids: Dict[str, int],  # 标签名称到id的映射字典
                 unk_token='<UNK>',
                 pad_token='<PAD>'
                 ):
        super().__init__()
        self._token2ids = token2ids
        self._unk_token_id = self._token2ids[unk_token]
        self._pad_token_id = self._token2ids[pad_token]
        self._label2ids = label2ids

    def __call__(self, text: str, label: Optional[str] = None) -> TokenizerOutput:
        # 1. 分词
        tokens = split_text_to_tokens(text)

        # 2. 将每个token转换为token id
        token_ids = [self._token2ids.get(token, self.unk_token_id) for token in tokens]

        # 3. 标签转换
        label_id = None
        if label is not None:
            label = str(label)
            label_id = self._label2ids[label]

        return TokenizerOutput(text, tokens, token_ids, label, label_id)

    @property
    def vocab_size(self):
        return len(self._token2ids)

    @property
    def num_classes(self):
        return len(self._label2ids)

    @property
    def pad_token_id(self):
        return self._pad_token_id

    @property
    def unk_token_id(self):
        return self._unk_token_id

    @property
    def token2ids(self):
        return self._token2ids

    @property
    def label2ids(self):
        return self._label2ids


class ProxyBertTokenizer(TokenizerBase):
    def __init__(self,
                 bert_tokenizer_file: str,
                 label2ids: Dict[str, int],  # 标签名称到id的映射字典
                 ):
        super().__init__()

        self.proxy: BertTokenizer = BertTokenizer.from_pretrained(bert_tokenizer_file)
        self._label2ids = label2ids

    def __call__(self, text: str, label: Optional[str] = None) -> TokenizerOutput:
        # 1. 分词
        tokens = self.proxy.tokenize(text)

        # 2. 将每个token转换为token id
        token_ids = self.proxy(text)['input_ids']

        # 3. 标签转换
        label_id = None
        if label is not None:
            label = str(label)
            label_id = self.label2ids[label]

        return TokenizerOutput(text, tokens, token_ids, label, label_id)

    @property
    def vocab_size(self):
        return self.proxy.vocab_size

    @property
    def num_classes(self):
        return len(self._label2ids)

    @property
    def pad_token_id(self):
        return self.proxy.pad_token_id

    @property
    def unk_token_id(self):
        return self.proxy.unk_token_id

    @property
    def token2ids(self):
        return dict(self.proxy.vocab.copy())

    @property
    def label2ids(self):
        return self._label2ids

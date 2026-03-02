# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/24 21:15
Create User : 19410
Desc : 数据相关的工具类方法
"""
from typing import List

import torch
from torch.utils.data import DataLoader

import jieba


# noinspection PyBroadException
def split_text_to_tokens_with_jieba(text: str) -> List[str]:
    try:
        return jieba.lcut(text)
    except Exception as e:
        print(f"异常: [{text}]")
        return []


def split_text_to_tokens_with_char(text: str) -> List[str]:
    return list(text)


def split_text_to_tokens(text: str) -> List[str]:
    """
    将给定文本转换为token列表
    :param text: 文本字符串, eg: 从这里怎么回家
    :return: token组成的list列表, eg: ["从", "这里", "怎么", "回家"]
    """
    _tokens = split_text_to_tokens_with_jieba(text)
    # _tokens = split_text_to_tokens_with_char(text)
    _tokens = [_token.lower() for _token in _tokens]
    return _tokens


def build_collect_fn(pad_token_id):
    def _collect_fn(_batch):
        # 获取当前批次中的最长序列的长度
        max_len = max([len(_item['token_ids']) for _item in _batch])
        # 合并
        _batch_text, _batch_tokens, _batch_token_ids, _batch_token_masks, _batch_label, _batch_label_id = [], [], [], [], [], []
        for _item in _batch:
            _batch_text.append(_item['text'])
            _batch_tokens.append(_item['tokens'])
            _batch_label.append(_item['label'])
            _batch_label_id.append(_item['label_id'])

            _token_ids = _item['token_ids']
            _token_masks = _item['token_masks']
            if len(_token_ids) < max_len:
                _pad_size = max_len - len(_token_ids)
                _token_ids = torch.cat([
                    _token_ids,
                    torch.ones(size=(_pad_size,), dtype=_token_ids.dtype, device=_token_ids.device) * pad_token_id
                ], dim=0)
                _token_masks = torch.cat([
                    _token_masks,
                    torch.zeros(size=(_pad_size,), dtype=_token_masks.dtype, device=_token_masks.device)
                ], dim=0)
            _batch_token_ids.append(_token_ids)
            _batch_token_masks.append(_token_masks)

        return {
            'text': _batch_text,
            'tokens': _batch_tokens,
            'label': _batch_label,
            'label_id': torch.stack(_batch_label_id, dim=0),
            'token_ids': torch.stack(_batch_token_ids, dim=0),
            'token_masks': torch.stack(_batch_token_masks, dim=0)
        }

    return _collect_fn


def build_dataloader(ds, batch_size, shuffle=False):
    return DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=build_collect_fn(pad_token_id=ds.tokenizer.pad_token_id)
    )

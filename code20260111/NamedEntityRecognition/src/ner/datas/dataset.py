# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/10 16:28
Create User : 19410
Desc : xxx
"""
import copy
import json
from typing import Dict

import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import Tokenizer


class NerTokenClassifyDataset(Dataset):
    def __init__(self,
                 in_file: str, tokenizer: Tokenizer, label2id: Dict[str, int],
                 append_special_token=False,
                 max_length=None
                 ):
        super().__init__()

        self.tokenizer: Tokenizer = tokenizer
        self.label2id: Dict[str, int] = label2id

        self.datas = []
        with open(in_file, "r", encoding="utf-8") as reader:
            for line in reader:
                obj = json.loads(line.strip())  # 将每行数据转换为json对象(字典)

                # 原始文本
                text = obj['originalText']

                # 标签数据的加载
                token_label_names = ['Other'] * len(text)
                for entity in obj['entities']:
                    label_type = entity['label_type']
                    start_pos = entity['start_pos']  # 包含
                    end_pos = entity['end_pos']  # 不包含
                    entity_len = end_pos - start_pos
                    if entity_len == 1:
                        token_label_names[start_pos] = f'S-{label_type}'
                    elif entity_len > 1:
                        token_label_names[start_pos] = f'B-{label_type}'
                        token_label_names[end_pos - 1] = f'E-{label_type}'
                        for i in range(start_pos + 1, end_pos - 1):
                            token_label_names[i] = f'M-{label_type}'

                # 分词处理
                token_iter = self.tokenizer(
                    text,
                    append_sep=append_special_token,
                    append_cls=append_special_token,
                    token_label_names=token_label_names,
                    no_entity_label_name='Other',
                    max_length=max_length
                )

                for data in token_iter:
                    label_names = data['label_names']
                    label_ids = [self.label2id[label_name] for label_name in label_names]
                    label_ids = torch.tensor(label_ids, dtype=torch.int64)
                    data['label_ids'] = label_ids
                    self.datas.append(data)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        return copy.deepcopy(self.datas[item])


# noinspection DuplicatedCode
def build_collect_fn(pad_token_id):
    def _collect_fn(_batch):
        # 获取当前批次中的最长序列的长度
        max_len = max([len(_item['token_ids']) for _item in _batch])
        # 合并
        _label_ids = None
        _batch_text, _batch_tokens, _batch_token_ids, _batch_token_masks, _batch_label_names, _batch_label_ids = [], [], [], [], [], []
        for _item in _batch:
            _batch_text.append(_item['text'])
            _batch_tokens.append(_item['tokens'])
            _batch_label_names.append(_item['label_names'])

            _token_ids = _item['token_ids']
            _token_masks = _item['token_masks']
            if 'label_ids' in _item:
                _label_ids = _item['label_ids']
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
                if _label_ids is not None:
                    _label_ids = torch.cat([
                        _label_ids,
                        torch.ones(size=(_pad_size,), dtype=_label_ids.dtype, device=_label_ids.device) * -100
                    ], dim=0)
            _batch_token_ids.append(_token_ids)
            _batch_token_masks.append(_token_masks)
            _batch_label_ids.append(_label_ids)

        return {
            'text': _batch_text,
            'tokens': _batch_tokens,
            'label_names': _batch_label_names,
            'label_ids': torch.stack(_batch_label_ids, dim=0) if _label_ids is not None else None,
            'token_ids': torch.stack(_batch_token_ids, dim=0),
            'token_masks': torch.stack(_batch_token_masks, dim=0)
        }

    return _collect_fn


def build_dataloader(ds: NerTokenClassifyDataset, batch_size, shuffle=False):
    return DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=build_collect_fn(pad_token_id=ds.tokenizer.pad_token_id)
    )

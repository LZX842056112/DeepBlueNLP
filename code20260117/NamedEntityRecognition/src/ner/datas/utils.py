# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/17 16:01
Create User : 19410
Desc : xxx
"""


def parse_record(text, entities, tokenizer, append_special_token, max_length, label2id, return_pt=True):
    # 标签数据的加载
    if entities is None:
        token_label_names = None
    else:
        token_label_names = ['Other'] * len(text)
        for entity in entities:
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
    token_iter = tokenizer(
        text,
        append_sep=append_special_token,
        append_cls=append_special_token,
        token_label_names=token_label_names,
        no_entity_label_name='Other',
        max_length=max_length,
        return_pt=return_pt
    )

    for data in token_iter:
        if entities is None:
            label_ids = None
        else:
            label_names = data['label_names']
            label_ids = [label2id[label_name] for label_name in label_names]
            if return_pt:
                import torch
                label_ids = torch.tensor(label_ids, dtype=torch.int64)
        data['label_ids'] = label_ids
        yield data

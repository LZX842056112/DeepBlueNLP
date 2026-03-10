# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/30 20:51
Create User : 19410
Desc : Bert作为解码器如何构造
"""

import torch

from transformers import BertConfig, BertLMHeadModel

if __name__ == '__main__':
    cfg = BertConfig(
        is_decoder=True,
        vocab_size=33,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
    )
    bert = BertLMHeadModel(cfg)
    print(bert)

    batch = {
        'input_ids': torch.tensor(
            [[2, 7, 19, 8, 7, 30, 30, 7, 26, 7, 3, 0, 0],
             [2, 15, 10, 11, 21, 13, 24, 7, 22, 14, 15, 9, 3],
             [2, 22, 11, 24, 27, 20, 3, 0, 0, 0, 0, 0, 0],
             [2, 9, 7, 24, 18, 7, 29, 3, 0, 0, 0, 0, 0]]
        ),
        'text': ['[CLS] ambaxxata [SEP]', '[CLS] ideographic [SEP]', '[CLS] perun [SEP]', '[CLS] carlaw [SEP]'],
        'attention_mask': torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]]
        ),
        'labels': torch.tensor(
            [[2, 7, 19, 8, 7, 30, 30, 7, 26, 7, 3, -100,
              -100],
             [2, 15, 10, 11, 21, 13, 24, 7, 22, 14, 15, 9,
              3],
             [2, 22, 11, 24, 27, 20, 3, -100, -100, -100, -100, -100,
              -100],
             [2, 9, 7, 24, 18, 7, 29, 3, -100, -100, -100, -100,
              -100]]
        )}

    bert_output = bert(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels'],
        use_cache=False,
        return_dict=True,
        output_attentions=True
    )
    print(bert_output.loss)
    print("nihao")

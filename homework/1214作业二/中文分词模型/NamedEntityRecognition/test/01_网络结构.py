# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/10 14:47
Create User : 19410
Desc : xxx
"""

import sys
import os

import torch

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

print(sys.path)


def t0():
    from ner.models.bert_ner import BertNerNetwork

    net = BertNerNetwork(
        bert_path=r"D:\huggingface\huggingface\hub\models--bert-base-chinese",
        num_classes=1 + 4 * 6
    )
    print(net)

    # input_ids = torch.tensor(
    #     [
    #         [12, 680, 1322, 0, 0, 0],
    #         [12, 680, 1322, 42, 38, 106]
    #     ],
    #     dtype=torch.int32
    # )
    # input_masks = torch.tensor([
    #     [1, 1, 1, 0, 0, 0],
    #     [1, 1, 1, 1, 1, 1]
    # ], dtype=torch.float32)
    #
    # score = net(input_ids, input_masks)
    # print(score)

    token_ids = torch.tensor([
        [
            101, 2642, 5442, 1728, 5395, 2642, 107, 5517, 4617, 107, 1762,
            2769, 7368, 750, 6121, 1059, 7937, 677, 5517, 4617, 3418, 3780, 3318, 102
        ]
    ])
    label_ids = torch.tensor([
        [
            0, 0, 0, 0, 0, 0, 0, 13, 15, 0, 0, 0,
            0, 0, 0, 9, 10, 10, 10, 10, 10, 10, 11, 0,
        ]
    ])
    token_masks = torch.ones_like(token_ids, dtype=torch.float32)
    score = net(token_ids, token_masks)
    print(token_ids.shape)
    print(label_ids.shape)
    print(score.shape)


def t1():
    from ner.models.bilstm_ner import BiLSTMNerNetwork

    net = BiLSTMNerNetwork(
        vocab_size=10000,
        hidden_size=64,
        num_classes=1 + 4 * 2,
        num_layers=3
    )
    print(net)

    input_ids = torch.tensor(
        [
            [12, 680, 1322, 0, 0, 0],
            [12, 680, 1322, 42, 38, 106]
        ],
        dtype=torch.int32
    )
    input_masks = torch.tensor([
        [1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1]
    ], dtype=torch.float32)

    score = net(input_ids, input_masks)
    print(score)


if __name__ == '__main__':
    t0()

# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/30 20:30
Create User : 19410
Desc : 网络构造：输入[N,T] 输出[N,T,33]
"""

import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Network, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False,  # 一定为False
            num_layers=3
        )
        self.classify = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask, labels=None):
        """
        :param input_ids: [N,T]
        :param attention_mask: [N,T]
        :param labels: [N,T]
        :return:
        """
        # 1. 获取输入token对应的特征向量 [N,T,E]
        feature_embed = self.embedding_layer(input_ids)

        # 2. 进一步的提取特征向量 [N,T,E]
        feature_embed, _ = self.lstm(feature_embed)

        # 3. 针对每个token进行分类决策判断，得到置信度 [N,T,vocab_size]
        logits = self.classify(feature_embed)

        # 4. 如果给定标签的情况下，进行损失计算
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = self.loss_fn(shift_logits.reshape(-1, self.vocab_size), shift_labels.reshape(-1))
        return logits, loss


def t0():
    net = Network(vocab_size=33, hidden_size=128)

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

    logits, loss = net(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels']
    )

    print(logits)
    print(logits.shape)
    print(loss)


if __name__ == '__main__':
    t0()

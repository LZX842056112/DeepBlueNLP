# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/10 14:56
Create User : 19410
Desc : 双向LSTM的Ner模型
"""
from typing import Optional

import torch
import torch.nn as nn

from .base import TokenClassifyNetwork


class BiLSTMNerNetwork(TokenClassifyNetwork):
    network_type: str = "lstm"

    def __init__(self, vocab_size, hidden_size, num_classes, num_layers=3):
        super().__init__()

        self.emb_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_size * 2,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            ) for i in range(num_layers)
        ])

        self.classify_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, 4 * hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(4 * hidden_size, num_classes)
        )

    def extract_token_features(self, token_ids, token_masks):
        # 1. embedding得到token向量 [bs,t,e]
        token_embs = self.emb_layer(token_ids)
        # 2. lstm的迭代特征向量的提取 [bs,t,2*e]
        input_embs = torch.concat([token_embs, token_embs], dim=-1)
        for lstm in self.lstm_layers:
            lstm_output, _ = lstm(input_embs)
            input_embs = input_embs + lstm_output
        return input_embs

    def classify_scores(self, token_embs):
        return self.classify_layer(token_embs)

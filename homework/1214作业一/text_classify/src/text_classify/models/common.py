# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/24 21:25
Create User : 19410
Desc : 定义模型相关内容
"""
import torch
import torch.nn as nn


class LSTMTextClassifyNetwork(nn.Module):
    def __init__(self, vocab_size, num_classes, hidden_size=128):
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)

        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_size, hidden_size=hidden_size,
                batch_first=True, bidirectional=False,
                num_layers=1
            ),
            nn.LSTM(
                input_size=hidden_size, hidden_size=hidden_size,
                batch_first=True, bidirectional=False,
                num_layers=1
            ),
            nn.LSTM(
                input_size=hidden_size, hidden_size=hidden_size,
                batch_first=True, bidirectional=False,
                num_layers=1
            ),
        ])

        self.classify = nn.Linear(hidden_size, num_classes, bias=False)

    def forward(self, token_ids, token_masks):
        """
        文本分类模型的前向执行过程
            bs: batch-size 样本数目
            T: 序列长度
        :param token_ids: 输入的样本token id tensor对象，shape形状为: [bs,T]; LongTensor; bs个文本，每个文本有T个token id； PS：由于文本的实际长度不一样，所以可能存在填充
        :param token_masks: 输入的token对应的填充信息，实际token id位置为1，填充位置为0，shape为:[bs,T]
        :return: [bs,num_classes] 针对每个样本输入当前样本属于各个类别的置信度值
        """
        # 1. 将token id转换为token向量 [bs,T] --> [bs,T,e]
        token_embs = self.embedding_layer(token_ids)

        # 2. 进一步提取更高阶的token向量 [bs,T,e] --> [bs,T,e]
        for lstm_layer in self.lstm_layers:
            lstm_output, _ = lstm_layer(token_embs)
            token_embs = token_embs + lstm_output

        # 3. 将token向量合并成文本向量 [bs,T,e] --> [bs,e]
        # text_embs = torch.mean(token_embs, dim=1)
        text_lens = torch.sum(token_masks, dim=1, keepdim=True)  # [bs,T] --> [bs,1]
        token_embs = token_embs * token_masks[:, :, None]  # [bs,T,e] * [bs,T,1] --> [bs,T,e]
        text_embs = torch.sum(token_embs, dim=1) / (text_lens + 1e-8)

        # 4. 基于文本特征向量进行全连接决策得到预测置信度 [bs,e] --> [bs,num_classes]
        score = self.classify(text_embs)

        return score

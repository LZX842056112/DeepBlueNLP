# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/24 21:25
Create User : 19410
Desc : 定义模型相关内容
"""
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import BertModel

from ..config import Config


class LSTMTextClassifyNetwork(nn.Module):
    network_type: str = 'lstm'

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


# noinspection DuplicatedCode
class BertTextClassifyNetwork(nn.Module):
    network_type: str = 'bert'

    def __init__(self, bert_path, num_classes, freeze: Optional[Union[bool, int]] = None):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path, add_pooling_layer=False, weights_only=False)
        # self.bert.encoder.layer = self.bert.encoder.layer[:3]  # 仅使用前3层

        if freeze is not None:
            if isinstance(freeze, bool):
                if freeze:
                    # 需要冻结bert的所有参数
                    for name, param in self.bert.named_parameters():
                        param.requires_grad = False
                        print(f"冻结参数:{name}")
            elif isinstance(freeze, int) and freeze > 0:
                # 冻结前多少层(EncoderLayer层)的参数
                freeze_layers = ["embeddings"]
                for layer_idx in range(freeze):
                    freeze_layers.append(f"encoder.layer.{layer_idx}.")
                for name, param in self.bert.named_parameters():
                    for freeze_layer_prefix in freeze_layers:
                        if name.startswith(freeze_layer_prefix):
                            param.requires_grad = False
                            print(f"冻结参数:{name}")
                            break

        self.classify = nn.Linear(self.bert.config.hidden_size, num_classes, bias=False)

    def forward(self, token_ids, token_masks):
        """
        文本分类模型的前向执行过程
            bs: batch-size 样本数目
            T: 序列长度
        :param token_ids: 输入的样本token id tensor对象，shape形状为: [bs,T]; LongTensor; bs个文本，每个文本有T个token id； PS：由于文本的实际长度不一样，所以可能存在填充
        :param token_masks: 输入的token对应的填充信息，实际token id位置为1，填充位置为0，shape为:[bs,T]
        :return: [bs,num_classes] 针对每个样本输入当前样本属于各个类别的置信度值
        """
        # 1. 调用bert得到bert的最终一层输出特征向量
        bert_output = self.bert(
            input_ids=token_ids,
            attention_mask=token_masks
        )
        last_hidden_state = bert_output[0]  # [bs,T,e]

        # 2. 将token向量合并成文本向量 [bs,T,e] --> [bs,e]
        text_embs = last_hidden_state[:, 0]  # 也就是获取[CLS]这个token对应的特征向量

        # 4. 基于文本特征向量进行全连接决策得到预测置信度 [bs,e] --> [bs,num_classes]
        score = self.classify(text_embs)

        return score


def build_network(config: Config):
    network_type = config.network_type

    if network_type == 'bert':
        return BertTextClassifyNetwork(
            bert_path=config.bert_path,
            num_classes=config.tokenizer.num_classes,
            freeze=config.freeze
        )
    else:
        return LSTMTextClassifyNetwork(
            vocab_size=config.tokenizer.vocab_size,
            num_classes=config.tokenizer.num_classes,
            hidden_size=config.hidden_size
        )

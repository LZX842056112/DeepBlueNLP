# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/10 14:35
Create User : 19410
Desc : xxx
"""
from typing import Optional, Union

import torch.nn as nn
from transformers import BertModel

from .base import TokenClassifyNetwork


class BertNerNetwork(TokenClassifyNetwork):
    network_type: str = "bert"

    def __init__(self, bert_path, num_classes, freeze: Optional[Union[bool, int]] = None):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_path, weights_only=False)

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

        _hidden_size: int = self.bert.config.hidden_size
        self.classify_layer = nn.Sequential(
            nn.Linear(_hidden_size, 4 * _hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(4 * _hidden_size, num_classes)
        )

    def extract_token_features(self, token_ids, token_masks):
        bert_output = self.bert(
            input_ids=token_ids,
            attention_mask=token_masks
        )
        # 获取最后一层的输出特征向量
        last_hidden_state = bert_output[0]  # [bs,T,e]
        return last_hidden_state

    def classify_scores(self, token_embs):
        return self.classify_layer(token_embs)

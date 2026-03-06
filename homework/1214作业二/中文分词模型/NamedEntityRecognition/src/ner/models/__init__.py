# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/10 14:27
Create User : 19410
Desc : xxx
"""

from .base import TokenClassifyNetwork
from .bert_ner import BertNerNetwork
from .bilstm_ner import BiLSTMNerNetwork
from ..config import Config


def build_network(config: Config):
    net_type = config.network_type.lower()
    if net_type == 'lstm':
        return BiLSTMNerNetwork(
            vocab_size=config.vocab_size, hidden_size=config.lstm_hidden_size,
            num_classes=config.num_classes, num_layers=config.lstm_layers
        )
    else:
        return BertNerNetwork(
            bert_path=config.bert_path,
            num_classes=config.num_classes,
            freeze=config.freeze
        )

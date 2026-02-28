# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/24 21:11
Create User : 19410
Desc : 相关的工具类
"""
import json
import os

import torch.nn as nn
import torch.optim as optim


def load_json(json_file):
    with open(json_file, "r", encoding="utf-8") as reader:
        return json.load(reader)


# noinspection PyTypeChecker
def save_json(json_file, json_obj):
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    with open(json_file, "w", encoding="utf-8") as writer:
        json.dump(json_obj, writer, indent=2, ensure_ascii=False)


def build_losses():
    return nn.CrossEntropyLoss()


def build_optim(net: nn.Module, lr):
    return optim.SGD(params=net.parameters(), lr=lr)

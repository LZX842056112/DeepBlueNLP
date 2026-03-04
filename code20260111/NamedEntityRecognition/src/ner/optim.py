# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/11 11:39
Create User : 19410
Desc : 优化器的定义
"""

import torch.nn as nn
import torch.optim as optim


def build_optim(net: nn.Module, lr: float):
    return optim.SGD(net.parameters(), lr)

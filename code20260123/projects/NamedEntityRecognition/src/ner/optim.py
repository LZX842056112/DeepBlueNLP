# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/11 11:39
Create User : 19410
Desc : 优化器的定义
"""
import torch.nn as nn
import torch.optim as optim


def build_optim(net: nn.Module, lr: float, lr_update_total_iters: int):
    opt = optim.SGD(net.parameters(), lr)
    lr_scheduler = optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=0.01,
        total_iters=lr_update_total_iters
    )

    return opt, lr_scheduler

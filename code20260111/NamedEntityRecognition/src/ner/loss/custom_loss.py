# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/11 11:33
Create User : 19410
Desc : xxx
"""
import torch
import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

    def forward(self, pred_score, target):
        """
        计算损失
        :param pred_score: 模型前向输出的置信度信息 [bs, t, num_classes]
        :param target: 实际标签id [bs, t] PS: 填充标签id为-100
        :return:
        """
        return self.loss2(pred_score, target)

    def loss1(self, pred_score, target):
        pred_score = torch.transpose(pred_score, dim0=2, dim1=1)
        loss = self.loss_fn(pred_score, target)

        # 获取所有属于实体的token损失
        ner_mask = (target > 0).to(dtype=loss.dtype)
        ner_loss = loss * ner_mask
        ner_loss = torch.sum(ner_loss) / (torch.sum(ner_mask) + 1e-8)

        # 获取所有非实体token的损失
        noner_mask = (target == 0).to(dtype=loss.dtype)
        noner_loss = loss * noner_mask
        noner_loss = torch.sum(noner_loss) / (torch.sum(noner_mask) + 1e-8)

        loss = torch.mean(loss)
        return loss, ner_loss, noner_loss

    def loss2(self, pred_score, target):
        """
        可以发现 target中绝大多数的token所属类别均是0(token不属于实体)
        :param pred_score:
        :param target:
        :return:
        """
        # 求解每个token的损失
        pred_score = torch.transpose(pred_score, dim0=2, dim1=1)
        loss = self.loss_fn(pred_score, target)  # [bs,t]

        # 获取所有属于实体的token损失
        ner_mask = (target > 0).to(dtype=loss.dtype)
        ner_loss = loss * ner_mask
        ner_loss = torch.sum(ner_loss) / (torch.sum(ner_mask) + 1e-8)

        # 获取所有非实体token的损失
        noner_mask = (target == 0).to(dtype=loss.dtype)
        k = max(int(torch.sum(ner_mask).item()) // 2, 1)
        noner_loss = loss * noner_mask
        noner_loss = noner_loss.view(-1)  # 扁平化
        noner_loss = torch.topk(noner_loss, k=k).values
        noner_loss = torch.mean(noner_loss)

        # 合并loss
        loss = ner_loss * 3.0 + noner_loss * 0.3
        return loss, ner_loss, noner_loss


def build_losses():
    return CustomLoss()

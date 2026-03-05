# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/10 14:27
Create User : 19410
Desc : xxx
"""
import torch
import torch.nn as nn


class TokenClassifyNetwork(nn.Module):
    export: bool = False
    network_type: str

    def __init__(self):
        super().__init__()

    def extract_token_features(self, token_ids, token_masks):
        """
        提取文本中的token对应的特征向量
        :param token_ids: [bs,t]
        :param token_masks: [bs,t]
        :return: [bs,t,e]
        """
        raise NotImplementedError("子类实现")

    def classify_scores(self, token_embs):
        """
        针对每个token的特征向量进行决策判断，得到每个token属于各个类别的置信度
        :param token_embs: [bs,t,e]
        :return: [bs,t,num_classes]
        """
        raise NotImplementedError("子类实现")

    def forward(self, token_ids, token_masks):
        """
        Token分类的前向过程
        :param token_ids: [N,T] LongTensor，N个样本，每个样本有T个token
        :param token_masks: [N,T] FloatTensor N个样本的每个token是否是填充token，0表示当前token是填充token，1表示实际token
        :return: [N,T,num_classes] N个样本，每个样本T个token，每个token对应num_classes个类别的置信度(概率)
        """
        # 1. 获取token的特征向量 [N,T,E]
        token_embs = self.extract_token_features(token_ids, token_masks)

        # 2. 获取每个token对应的类别预测置信度 [N,T,num_classes]
        score = self.classify_scores(token_embs)

        # 3. 返回结果
        if self.training:
            return score
        else:
            if self.export:
                return torch.softmax(score, dim=-1)
            return score, torch.softmax(score, dim=-1)

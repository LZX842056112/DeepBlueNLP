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


# 1. 基于 LSTM 的自定义文本分类网络
class LSTMTextClassifyNetwork(nn.Module):
    network_type: str = 'lstm'

    def __init__(self, vocab_size, num_classes, hidden_size=128):
        super().__init__()
        # 词嵌入层：将离散的 Token ID (如 1024) 映射成连续的稠密向量 (维度为 hidden_size)
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)

        # 使用 ModuleList 堆叠 3 层独立的 LSTM。
        # 为什么不直接用 num_layers=3？因为作者在 forward 中想做“残差连接”！
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_size, hidden_size=hidden_size,
                batch_first=True, bidirectional=False,  # batch_first=True 表示输入的 shape 是 [batch, seq, feature]
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

        # 分类头：全连接层，把文本的特征向量映射到具体的类别数量上
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
        # 1. 查表：将 token id 转换为词向量。形状变化: [bs, T] --> [bs, T, hidden_size]
        token_embs = self.embedding_layer(token_ids)

        # 2. 进一步提取更高阶的token向量 [bs,T,e] --> [bs,T,e]
        # 序列特征提取：依次通过 3 层 LSTM
        for lstm_layer in self.lstm_layers:
            # LSTM 返回两个值：输出特征和隐状态。这里只取输出特征
            lstm_output, _ = lstm_layer(token_embs)
            # 💡 高级技巧：残差连接 (Residual Connection)。将输入和输出相加，防止梯度消失！
            token_embs = token_embs + lstm_output

        # 3. 将token向量合并成文本向量 [bs,T,e] --> [bs,e]
        # text_embs = torch.mean(token_embs, dim=1)
        # 将序列变长特征池化为定长特征 (Masked Mean Pooling)
        # 算出每个句子真实的有效长度 (把掩码里的 1 加起来)
        text_lens = torch.sum(token_masks, dim=1, keepdim=True)  # [bs,T] --> [bs,1]
        # 将 Padding 位置的向量强行置为 0，防止 Padding 干扰语义
        # token_masks[:, :, None] 是为了在最后增加一个维度变成 [bs, T, 1] 以便触发广播机制
        token_embs = token_embs * token_masks[:, :, None]  # [bs, T, e] * [bs, T, 1] --> [bs, T, e]

        # 将所有有效字的向量按时间维度(dim=1)相加，然后除以有效长度求平均。+1e-8 是为了防止除以 0 导致报错
        text_embs = torch.sum(token_embs, dim=1) / (text_lens + 1e-8)  # [bs, e]

        # 4. 基于文本特征向量进行全连接决策得到预测置信度 [bs,e] --> [bs,num_classes]
        score = self.classify(text_embs)

        return score


# 2. 基于 BERT 微调的文本分类网络
# noinspection DuplicatedCode
class BertTextClassifyNetwork(nn.Module):
    network_type: str = 'bert'

    def __init__(self, bert_path, num_classes, freeze: Optional[Union[bool, int]] = None):
        super().__init__()
        # 加载预训练的 BERT 模型
        # add_pooling_layer=False: 我们不需要 BERT 自带的针对 [CLS] 的额外全连接池化层，我们自己实现分类头
        self.bert = BertModel.from_pretrained(bert_path, add_pooling_layer=False, weights_only=False)
        # self.bert.encoder.layer = self.bert.encoder.layer[:3]  # 仅使用前3层

        # 💡 高级技巧：模型参数冻结 (Freezing)
        if freeze is not None:
            if isinstance(freeze, bool):
                # 如果 freeze 为 True，冻结整个 BERT。意味着 BERT 退化成一个固定的“词向量提取器”，只训练分类头
                if freeze:
                    # 需要冻结bert的所有参数
                    for name, param in self.bert.named_parameters():
                        param.requires_grad = False
                        print(f"冻结参数:{name}")
            elif isinstance(freeze, int) and freeze > 0:
                # 冻结前多少层(EncoderLayer层)的参数
                # 如果 freeze 是一个数字 (例如 3)，则冻结 embedding 层和前 3 层 Transformer Encoder
                freeze_layers = ["embeddings"]
                for layer_idx in range(freeze):
                    freeze_layers.append(f"encoder.layer.{layer_idx}.")
                for name, param in self.bert.named_parameters():
                    for freeze_layer_prefix in freeze_layers:
                        # 使用前缀匹配来锁定具体的层
                        if name.startswith(freeze_layer_prefix):
                            param.requires_grad = False
                            print(f"冻结参数:{name}")
                            break
        # 分类头：接收 BERT 输出的 hidden_size (一般是 768)，映射到类别数
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
        # 取出最后一层的隐藏状态 [bs, T, 768]
        last_hidden_state = bert_output[0]  # [bs,T,e]

        # 2. 提取分类特征, 将token向量合并成文本向量 [bs,T,e] --> [bs,e]
        # 💡 BERT 的特殊设计：序列的第一个 token 是 [CLS]。
        # 经过注意力机制，[CLS] 的向量已经汇聚了整句话的全局语义信息，所以直接取第 0 个位置的向量做分类即可！
        text_embs = last_hidden_state[:, 0]  # 也就是获取[CLS]这个token对应的特征向量

        # 4. 基于文本特征向量进行全连接决策得到预测置信度 [bs,e] --> [bs,num_classes]
        score = self.classify(text_embs)

        return score


# 3. 工厂模式：根据配置构建网络
def build_network(config: Config):
    """
    一个简单的工厂函数，解耦了模型实例化与配置解析。
    这样主训练脚本里只需要调 build_network(config) 即可，不用写一堆 if-else。
    """
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

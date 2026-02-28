# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/21 10:32
Create User : 19410
Desc : xxx
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def t0():
    """
    文本处理的通用简化过程：从自然语言文本到稠密特征向量(Embedding)
    :return:
    """
    # 构建词表，将汉字/词组映射为唯一的数字ID
    token2idx = {
        '从': 23, '这里': 58, '怎么': 102, '回家': 203,
    }
    # 初始化词嵌入层：词表大小为300，每个词映射为8维的向量
    embedding_layer = nn.Embedding(num_embeddings=300, embedding_dim=8)

    # 1. 最原始数据
    text = "从这里怎么回家"

    # 2. 分词
    tokens = ['从', '这里', '怎么', '回家']

    # 3. 基于构建好的单词id映射mapping将每个token转换为id
    # bs(batch_size)=1，序列长度(seq_len)=4
    token_ids = [[23, 58, 102, 203]]

    # 4. token id做embedding得到对应的token特征向量
    # 输入维度: [bs=1, t=4] --> 输出维度: [bs=1, t=4, e=8] (e为特征维度)
    # [bs,t,e] --> bs个样本，每个样本由t个token组成，每个token对应的稠密特征向量的维度大小为e
    token_embs = embedding_layer(torch.tensor(token_ids))
    print(token_embs.shape)

    # 5. 接下来的网络结构应该就是基于token的特征向量进一步的进行特征的组合/提取 --> 最终得到文本向量 ---> 基于文本向量进行最终的决策输出，得到属于各个类别的置信度信息


def tt_with_fc():
    """
    基于全连接提取token特征向量：
        问题：token之间没有进行特征的交互，token特征向量的提取仅基于当前token，没有考虑上下文
    :return:
    """
    bs, t, e = 1, 5, 128
    dtype = torch.float32
    # 定义全连接层，将128维映射到256维
    linear = nn.Linear(e, e * 2, dtype=dtype)
    token_embs = torch.randn(bs, t, e, dtype=dtype)  # 一般为上一个模块的输出特征向量

    # 5. 全连接提取特征向量
    # 线性层只会作用在最后一个维度(e)上
    # [bs, t, e] * 权重矩阵[e, 2e] + 偏置[2e] --> [bs, t, 2e]
    token_new_embs = linear(token_embs)  # [bs,t,e] * [e,2e] + [2e] --> [bs,t,2e]
    print(token_new_embs.shape)


def tt_with_conv1d():
    """
    基于1D卷积(Conv1D)提取特征：
    进步与问题：通过滑动窗口(kernel_size=3)，当前词能和前后的词进行特征融合。
    但依然只能做“局部”融合，遇到长句子时，首尾的词很难建立联系（需要堆叠很多层卷积）。
    """
    bs, t, e = 1, 5, 128
    dtype = torch.float32
    # in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
    conv1d = nn.Conv1d(e, 2 * e, 3, 1, padding=1, dtype=dtype)
    token_embs = torch.randn(bs, t, e, dtype=dtype)  # 一般为上一个模块的输出特征向量

    # 5. 提取特征向量
    # 1D卷积要求序列维度在最后，特征通道在中间，所以需要调换维度
    token_embs_t = torch.permute(token_embs, dims=(0, 2, 1))  # [bs,t,e] --> [bs,e,t]
    token_new_embs = conv1d(token_embs_t)  # [bs,e,t] --> [bs,e,t]
    # 计算完成后再把维度调换回来
    token_new_embs = torch.permute(token_new_embs, dims=(0, 2, 1))  # [bs,e,t] --> [bs,t,e]
    print(token_new_embs.shape)


def tt_with_rnn03():
    """
    基于循环神经网络(RNN)提取特征：
    进步：通过隐藏状态的传递，后面的词能够获取前面所有词的信息，实现了“全局”融合。
    """
    bs, t, e = 1, 5, 128
    dtype = torch.float32
    token_embs = torch.randn(bs, t, e, dtype=dtype)  # 一般为上一个模块的输出特征向量
    rnn = nn.RNN(
        input_size=e,  # 每个时刻的输入特征向量维度
        hidden_size=2 * e,  # 每个时刻期望输出的特征向量维度大小，也就是细胞信息的维度大小
        num_layers=3,  # RNN的层数
        nonlinearity='tanh',  # 激活函数，当前API仅支持tanh和relu
        bias=True,  # 内部线性转换的时候是否有bias操作
        batch_first=True,  # 输入数据中bs这个维度是第一维还是第二维(是不是在最前面)，True表示shape为[bs,t,e]; False表示[t,bs,e]
        dropout=0.0,
        bidirectional=True,  # 是否是双向的RNN结构
        device=None,
        dtype=dtype
    )
    for name, param in rnn.named_parameters():
        print(name, "---->", param.shape)

    # 5. 提取特征向量
    # rnn_output: [bs,t,v] RNN输出的每个样本、每个token对应的v维的特征向量(如果是多层rnn结构，仅最后一层输出)
    # rnn_state:[?,bs,v]  序列最后一个时刻/token对应的状态信息 --> 在RNN中，就是和最后一个时刻的输出值一样
    ##### ?表示的是 ?=(2 if bidirectional else 1) * num_layers
    rnn_output, rnn_state = rnn(token_embs)
    print(rnn_output.shape)  # [1, 5, 512] (2*e=256，双向再乘2=512)
    print(rnn_state.shape)  # [6, 1, 256] (3层*双向=6)

    # 6. 获取整个句子的全局文本特征向量：通常将所有时间步的输出求均值
    text_emb = torch.mean(rnn_output, dim=1)
    print(text_emb.shape)


def qkv_attention_value(q, k, v, masks=None):
    """
    QKV (Query-Key-Value) Attention 核心计算公式
    支持普通 Attention 和 Multi-Head Attention 的维度
    :param q:  [bs, qt, e] or [bs, h, qt, e] bs个样本，每个样本qt个query，每个query是一个e维的向量
    :param k:  [bs, kvt, e] or [bs, h, kvt, e] bs个样本，每个样本kvt个key，每个key是一个e维的向量
    :param v:  [bs, kvt, e] or [bs, h, kvt, e] bs个样本，每个样本kvt个value，每个value是一个e维的向量
    :param masks: [bs, qt, kvt] or [bs, 1, qt, kvt] bs个样本，每个样本对应的mask矩阵，允许计算attention的位置为0，不允许的位置为负无穷
    :return: [bs, h, qt, e]
    """
    # 获取特征维度大小，用于后续的缩放(缩放点积注意力)
    e = q.shape[-1]

    # Attention计算
    # 1. 计算 Query 和 Key 之间的相关性得分
    # 单头: [bs, qt, e] * [bs, e, kvt] --> [bs, qt, kvt]
    # 多头: [bs, h, qt, e] * [bs, h, e, kvt] --> [bs, h, qt, kvt]
    # dim0=-1表示倒数第一维，dim1=-2表示倒数第二维，实现矩阵转置
    # bs个样本，每个样本有qt的query，每个query和kvt个key之间的相关性
    score = torch.matmul(q, torch.transpose(k, dim0=-1, dim1=-2))
    # 除以根号e，防止点积结果过大导致softmax进入梯度饱和区
    score = score / np.sqrt(e)  # [bs, qt, kvt] or [bs, h, qt, kvt]
    # 如果传入了掩码(Mask)，将不需要被注意到的地方(如填充词<PAD>)的值减去一个无穷大
    if masks is not None:
        score = score + masks

    # 2. 将相关性得分经过 Softmax 转换为总和为 1 的权重概率 (0~1之间)
    alpha = torch.softmax(score, dim=-1)  # [bs, qt, kvt] or [bs, h, qt, kvt]

    # 3. 加权合并：用算出的权重对 Value 进行加权求和
    # 单头: [bs, qt, kvt] * [bs, kvt, e] ---> [bs, qt, e]
    # 多头: [bs, h, qt, kvt] * [bs, h, kvt, e] ---> [bs, h, qt, e]
    value = torch.matmul(alpha, v)

    return value


def tt_with_self_attention01():
    bs, t, e = 1, 5, 128
    dtype = torch.float32
    token_embs = torch.randn(bs, t, e, dtype=dtype)  # 一般为上一个模块的输出特征向量

    wq_linear = nn.Linear(in_features=e, out_features=e, bias=False)
    wk_linear = nn.Linear(e, e, bias=False)
    wv_linear = nn.Linear(e, e, bias=False)

    # 5. Self-Attention 提取特征向量
    q = wq_linear(token_embs)  # [bs,t,e] * [e,e] --> [bs,t,e]
    k = wk_linear(token_embs)  # [bs,t,e] * [e,e] --> [bs,t,e]
    v = wv_linear(token_embs)  # [bs,t,e] * [e,e] --> [bs,t,e]
    token_new_embs = qkv_attention_value(q, k, v)
    print(token_new_embs.shape)


# 单头自注意力机制 (Self-Attention) 模块封装
class SelfAttentionModule(nn.Module):
    def __init__(self, hidden_size, in_features=None):
        super().__init__()
        if in_features is None:
            in_features = hidden_size

        # 定义生成 Q, K, V 的三个线性映射层
        self.wq_linear = nn.Linear(in_features=in_features, out_features=hidden_size, bias=False)
        self.wk_linear = nn.Linear(in_features, hidden_size, bias=False)
        self.wv_linear = nn.Linear(in_features, hidden_size, bias=False)

    def forward(self, x):
        """
        Self-Attention的计算
        :param x: 输入特征向量 [bs,t,hidden_size]
        :return:  输出特征向量 [bs,t,hidden_size]
        """
        # x 自己跟自己做 Attention，所以 Q, K, V 的来源都是 x 本身
        q = self.wq_linear(x)  # [bs,t,hidden_size] * [hidden_size,hidden_size] --> [bs,t,hidden_size]
        k = self.wk_linear(x)  # [bs,t,hidden_size] * [hidden_size,hidden_size] --> [bs,t,hidden_size]
        v = self.wv_linear(x)  # [bs,t,hidden_size] * [hidden_size,hidden_size] --> [bs,t,hidden_size]
        return qkv_attention_value(q, k, v)


def tt_with_self_attention02():
    bs, t, e = 1, 5, 128
    dtype = torch.float32
    token_embs = torch.randn(bs, t, e, dtype=dtype)  # 一般为上一个模块的输出特征向量

    atten_module = SelfAttentionModule(hidden_size=e)

    # 5. Self-Attention 提取特征向量
    token_new_embs = atten_module(token_embs)
    print(token_new_embs.shape)


class MultiHeadSelfAttentionModuleV0(nn.Module):
    def __init__(self, hidden_size, num_head):
        super().__init__()
        _hidden_size = hidden_size // num_head

        self.atten_list = nn.ModuleList([
            SelfAttentionModule(hidden_size=_hidden_size, in_features=hidden_size) for _ in range(num_head)
        ])
        self.wo = nn.Linear(
            in_features=_hidden_size * num_head,
            out_features=hidden_size
        )

    def forward(self, x):
        # 1. 多个self-attention分别提取特征向量
        atten_values = []
        for atten_module in self.atten_list:
            atten_value = atten_module(x)  # [bs,t,hidden_size] --> [bs,t,_hidden_size]
            atten_values.append(atten_value)
        # 2. 合并输出  [bs,t,_hidden_size * num_head]
        atten_value = torch.concat(atten_values, dim=-1)
        atten_value = self.wo(atten_value)
        return atten_value


# 多头自注意力机制 (Multi-Head Self-Attention) - 工业级矩阵并行实现版本
class MultiHeadSelfAttentionModuleV1(nn.Module):
    def __init__(self, hidden_size, num_head):
        super().__init__()
        # 计算每个"头"分到的特征维度 (例如 128维 4个头，每个头就是 32维)
        _hidden_size = hidden_size // num_head
        # 为了高效，我们依然用一个大的线性层一次性算出所有头的特征，之后再切分
        _qkv_hidden_size = _hidden_size * num_head

        self.num_head = num_head
        self.wq_linear = nn.Linear(in_features=hidden_size, out_features=_qkv_hidden_size, bias=False)
        self.wk_linear = nn.Linear(hidden_size, _qkv_hidden_size, bias=False)
        self.wv_linear = nn.Linear(hidden_size, _qkv_hidden_size, bias=False)

        # 最终合并所有头结果的输出线性层
        self.wo = nn.Linear(
            in_features=_qkv_hidden_size,
            out_features=hidden_size
        )

    def forward(self, x, token_masks=None):
        bs, t, _ = x.shape

        # 1. 多个self-attention分别提取特征向量
        # 1. 经过线性层映射，得到融合了所有头的 Q, K, V
        # _qkv_hidden_size = num_head * e
        q = self.wq_linear(x)  # [bs,t,hidden_size] * [hidden_size,_qkv_hidden_size] --> [bs,t,_qkv_hidden_size]
        k = self.wk_linear(x)  # [bs,t,hidden_size] * [hidden_size,_qkv_hidden_size] --> [bs,t,_qkv_hidden_size]
        v = self.wv_linear(x)  # [bs,t,hidden_size] * [hidden_size,_qkv_hidden_size] --> [bs,t,_qkv_hidden_size]
        # 2. 核心操作：将特征切分给多个"头" (变形操作)
        # 先按头的数量拆分特征维度: [bs, t, num_head, 每个头的维度e]
        q = q.reshape(bs, t, self.num_head, -1)  # [bs,t,_qkv_hidden_size] -> [bs,t,h,e]
        # 调换维度，把"头(h)"提前，方便后续并行矩阵乘法: [bs, h, t, e]
        q = torch.permute(q, dims=(0, 2, 1, 3))  # [bs,t,h,e] -> [bs,h,t,e]
        # K 和 V 同理
        k = k.reshape(bs, t, self.num_head, -1)  # [bs,t,_qkv_hidden_size] -> [bs,t,h,e]
        k = torch.permute(k, dims=(0, 2, 1, 3))  # [bs,t,h,e] -> [bs,h,t,e]
        v = v.reshape(bs, t, self.num_head, -1)  # [bs,t,_qkv_hidden_size] -> [bs,t,h,e]
        v = torch.permute(v, dims=(0, 2, 1, 3))  # [bs,t,h,e] -> [bs,h,t,e]
        # 3. 调用基础的 QKV 函数执行多头 Attention 计算
        atten_value = qkv_attention_value(q, k, v, masks=token_masks)  # [bs,h,t,e]
        # 4. 恢复形状并将多个头的结果重新拼接起来
        atten_value = torch.permute(atten_value, dims=(0, 2, 1, 3))  # [bs,h,t,e] -> [bs,t,h,e]
        atten_value = atten_value.reshape(bs, t, -1)  # [bs,t,h,e] --> [bs,t,_qkv_hidden_size]
        # 5. 最后过一层线性层融合信息 [bs,t,_hidden_size * num_head]
        atten_value = self.wo(atten_value)
        return atten_value


def tt_with_self_multiheadattention01():
    bs, t, e = 1, 5, 128
    dtype = torch.float32
    token_embs = torch.randn(bs, t, e, dtype=dtype)  # 一般为上一个模块的输出特征向量

    atten_module = MultiHeadSelfAttentionModuleV0(hidden_size=e, num_head=4)

    # 5. Self-Attention 提取特征向量
    token_new_embs = atten_module(token_embs)
    print(token_new_embs.shape)


def tt_with_self_multiheadattention02():
    bs, t, e = 1, 5, 128
    dtype = torch.float32
    token_embs = torch.randn(bs, t, e, dtype=dtype)  # 一般为上一个模块的输出特征向量

    atten_module = MultiHeadSelfAttentionModuleV1(hidden_size=e, num_head=4)

    # 5. Self-Attention 提取特征向量
    token_new_embs = atten_module(token_embs)
    print(token_new_embs.shape)


# 掩码 (Mask) 的应用场景演示：让模型忽略特定的词(如Padding或未来的词)
def tt_with_attention_masks():
    emb_layer = nn.Embedding(100, 32)
    atten_module = MultiHeadSelfAttentionModuleV1(hidden_size=32, num_head=4)

    # 用处一：编码器中使用 ---> 编码器中的填充
    # token_ids 中 0 通常代表 <PAD> (由于句子长度不同，用0补齐)
    token_ids = torch.tensor([
        [3, 15, 18, 21, 25, 22],  # 句子1：真实长度6
        [1, 8, 9, 10, 0, 0]  # 句子2：真实长度4，后面两个是填充
    ])
    # 构建掩码矩阵：1.0代表需要保留关注，0.0代表需要被忽略(屏蔽)
    # 对于句子2的最后两列，全是0，表示任何词都不应该关注这两个<PAD>词
    encoder_token_masks = torch.tensor([
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        ],
        [
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        ]
    ])
    # 【核心技巧】：将 0.0 变成负无穷(接近的极小值)，将 1.0 变成 0.0
    # 这样在 QKV 函数中执行 score + masks 时，不需要关注的地方分数就会变成负无穷
    # 经过 Softmax 后，概率就会变为绝对的 0。
    encoder_token_masks = (1.0 - encoder_token_masks) * torch.finfo(encoder_token_masks.dtype).min
    # 增加一个头的维度适配多头计算
    encoder_token_masks = encoder_token_masks[:, None]

    attent_value = atten_module(emb_layer(token_ids), token_masks=encoder_token_masks)
    print(attent_value.shape)

    # 用户二：解码器中使用 ---> self-attention只能计算单方向的相关性、cross-attention需要考虑编码器的填充
    # 解码器在生成文本时，是严格从左往右的。当前词绝对不能“偷看”到它后面的词。
    # 这是一个下三角矩阵：主对角线及左下全是 1 (允许关注自己和过去的词)，右上全是 0 (屏蔽未来的词)
    token_ids = torch.tensor([
        [3, 15, 18, 21, 25, 22],
        [1, 8, 9, 10, 0, 0]
    ])
    decoder_token_masks = torch.tensor([
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 第1个词只能看第1个词
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # 第2个词能看1, 2
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # 第3个词能看1, 2, 3
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ]
    ])
    # 同样的转换逻辑：将需要屏蔽的地方变成负无穷
    decoder_token_masks = (1.0 - decoder_token_masks) * torch.finfo(decoder_token_masks.dtype).min
    decoder_token_masks = decoder_token_masks[:, None]

    attent_value = atten_module(emb_layer(token_ids), token_masks=decoder_token_masks)
    print(attent_value.shape)


if __name__ == '__main__':
    tt_with_self_attention02()
    tt_with_self_multiheadattention01()
    tt_with_self_multiheadattention02()
    tt_with_attention_masks()

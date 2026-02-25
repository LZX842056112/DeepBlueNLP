# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/14 15:06
Create User : 19410
Desc : xxx
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def tt_with_lstm01():
    bs, t, e = 1, 10, 4
    v = 3
    lstm = nn.LSTM(
        input_size=e,  # 每个时刻/每个token对应的输入向量维度大小
        hidden_size=v,  # 期望每个时刻输出的向量维度大小
        num_layers=1,  # 层数
        bias=True,  # 是否添加bias
        batch_first=True,  # 批次是不是第一维， True表示[bs,t,e] False表示[t,bs,e]
        dropout=0.0,
        bidirectional=False  # 是否是双向的LSTM结构
    )
    print("LSTM内部的参数shape:")
    for name, param in lstm.named_parameters():
        print(name, "--->", param.shape)

    # 上一个模块的输出特征向量(Embedding模块)
    token_embs = torch.randn(bs, t, e)
    print(f"LSTM输入的特征向量维度:{token_embs.shape}")

    # 调用lstm
    lstm_output, (lstm_h_n, lstm_c_n) = lstm(token_embs)
    print(f"LSTM每个时刻的输出特征向量:\n\t{lstm_output.shape}")
    print(f"LSTM最后一个时刻的输出特征向量ht:\n\t{lstm_output[:, -1]}")
    print(f"LSTM最后一个时刻的输出特征向量ht:\n\t{lstm_h_n}")
    print(f"LSTM最后一个时刻的状态向量ct:\n\t{lstm_c_n}")


def tt_with_lstm02():
    """
    LSTM过程公式拆解
    :return:
    """
    bs, t, e = 1, 5, 4
    v = 3

    u_ft = nn.Parameter(torch.randn(e, v))
    w_ft = nn.Parameter(torch.randn(v, v))

    u_it = nn.Parameter(torch.randn(e, v))
    w_it = nn.Parameter(torch.randn(v, v))

    u_ct = nn.Parameter(torch.randn(e, v))
    w_ct = nn.Parameter(torch.randn(v, v))

    u_ot = nn.Parameter(torch.randn(e, v))
    w_ot = nn.Parameter(torch.randn(v, v))

    lstm = nn.LSTM(
        input_size=e,  # 每个时刻/每个token对应的输入向量维度大小
        hidden_size=v,  # 期望每个时刻输出的向量维度大小
        num_layers=1,  # 层数
        bias=False,  # 是否添加bias
        batch_first=True,  # 批次是不是第一维， True表示[bs,t,e] False表示[t,bs,e]
        dropout=0.0,
        bidirectional=False  # 是否是双向的LSTM结构
    )
    print("LSTM内部的参数shape:")
    for name, param in lstm.named_parameters():
        print(name, "--->", param.shape)

    u_it, u_ft, u_ct, u_ot = torch.split(lstm.weight_ih_l0.T, split_size_or_sections=v, dim=1)
    w_it, w_ft, w_ct, w_ot = torch.split(lstm.weight_hh_l0.T, split_size_or_sections=v, dim=1)

    # 4. token_ids哑编码 + 每个单词(哑编码的向量)进行全连接转换 每个token都得到一个稠密的项目
    # bs个样本，每个样本由t个token组成，每个token对应的稠密特征向量的维度大小为128
    # bs: batch size ; t: 时刻/序列长度; e: 每个token对应的向量维度大小
    token_embs = torch.rand(bs, t, e)  # [bs,t,e]

    # 5. 解决全连接的特征问题: 全连接提取特征的时候仅考虑当前时刻的token输入，不考虑序列的特征
    new_token_embs_list = []
    ht = torch.zeros((bs, v))
    ct = torch.zeros((bs, v))
    for _t in range(t):
        # 遗忘门
        ft = F.sigmoid(torch.matmul(token_embs[:, _t, :], u_ft) + torch.matmul(ht, w_ft))
        # 更新门
        it = F.sigmoid(torch.matmul(token_embs[:, _t, :], u_it) + torch.matmul(ht, w_it))
        cur_ct = F.tanh(torch.matmul(token_embs[:, _t, :], u_ct) + torch.matmul(ht, w_ct))
        # 输出门
        ot = F.sigmoid(torch.matmul(token_embs[:, _t, :], u_ot) + torch.matmul(ht, w_ot))

        # 更新当前时刻对应的状态信息
        ct = ct * ft + cur_ct * it

        # 获取当前输出
        ht = ot * F.tanh(ct)

        oi = ht[:, None]  # 增加一个维度 [bs,64] -> [bs,1,64]
        new_token_embs_list.append(oi)
    new_token_embs = torch.cat(new_token_embs_list, dim=1)
    print(new_token_embs.shape)

    # LSTM的结果
    lstm_output, (lstm_h_n, lstm_c_n) = lstm(token_embs)

    print(torch.max(torch.abs(new_token_embs - lstm_output)))


if __name__ == '__main__':
    # tt_with_lstm01()
    tt_with_lstm02()

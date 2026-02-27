# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/20 11:36
Create User : 19410
Desc : 最原始的结构：解码器每次仅解码一个token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(24)


class EncoderModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()

        self.embed_layer = nn.Embedding(
            num_embeddings=vocab_size,  # 词汇表大小 --> token到id转换的映射表大小
            embedding_dim=hidden_size  # 每个词/Token对应的特征向量维度大小
        )
        self.rnn_layer = nn.LSTM(
            input_size=hidden_size,  # 每个token输入的特征向量维度大小
            hidden_size=hidden_size,  # 每个token输出的特征向量维度大小
            num_layers=num_layers,  # 层数
            batch_first=True,
            bidirectional=True
        )
        # 将RNN输出特征值转换为高阶特征向量C
        self.ctx_feature_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, token_ids):
        # 1. token id转换为token embedding向量 [bs,et] -> [bs,et,hidden_size]
        token_embed = self.embed_layer(token_ids)
        # 2. 调用rnn结构获取序列特征向量
        # output: [bs,et,2*hidden_size] --> 当前是双向结构
        # state: rnn和gru的时候，只有一个值；lstm的时候，有两个值(二元组)；shape均为[?,bs,hidden_size]
        output, state = self.rnn_layer(token_embed)
        if isinstance(state, tuple):
            state = state[0] + state[1]
        state = torch.mean(state, dim=0)  # [?,bs,hidden_size] -->  [bs,hidden_size]
        # 3. 将状态信息转换为文本特征向量 [bs,hidden_size] -> [bs,hidden_size]
        ctx_embed = self.ctx_feature_layer(state)

        return ctx_embed  # [bs,e]


class DecoderModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers

        self.embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)

        self.rnn_init_h0_layers = nn.Linear(hidden_size, num_layers * hidden_size)
        self.rnn_init_c0_layers = nn.Linear(hidden_size, num_layers * hidden_size)

        self.rnn_layer = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True, bidirectional=False
        )
        self.classify_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids, encoder_ctx):
        """
            前向执行过程
            : token_ids : [bs,dt] 解码器输入token ids
            : encoder_ctx : [bs, hidden_size] 编码器提取出来的文本特征向量
        """
        # 1. 针对编码器提取特征向量进行特征提取，获取lstm的初始状态
        bs, e = encoder_ctx.shape
        h0 = self.rnn_init_h0_layers(encoder_ctx)  # [bs,hidden_size] -> [bs,num_layers*hidden_size]
        h0 = h0.reshape((bs, e, -1))  # [bs,num_layers*hidden_size] --> [bs,hidden_size,num_layers]
        h0 = torch.permute(h0, dims=(2, 0, 1))  # [bs,hidden_size,num_layers] -> [num_layers,bs,hidden_size]

        c0 = self.rnn_init_c0_layers(encoder_ctx)  # [bs,hidden_size] -> [bs,num_layers*hidden_size]
        c0 = c0.reshape((bs, e, -1))  # [bs,num_layers*hidden_size] --> [bs,hidden_size,num_layers]
        c0 = torch.permute(c0, dims=(2, 0, 1))  # [bs,hidden_size,num_layers] -> [num_layers,bs,hidden_size]

        if self.training:
            # 2. 针对输入数据进行embedding操作 [bs,dt] -> [bs,dt,hidden_size]
            token_embed = self.embed_layer(token_ids)

            # 3. 调用rnn结构获取序列特征向量
            # output: [bs,dt,hidden_size] 每个token对应的特征向量
            output, _ = self.rnn_layer(token_embed, (h0, c0))

            # 4. 针对每个token进行全连接得到预测对应类别 [bs,dt,hidden_size] -> [bs,dt,vocab_size]
            score = self.classify_layer(output)

            return score
        else:
            ##### 针对推理预测过程，需要一个token、一个token进行输入预测得到结果
            i = 0
            cur_token_ids = token_ids  # 当前时刻的token id
            hcn = (h0, c0)

            while True:
                # 2. 针对输入数据进行embedding操作 [bs,dt] -> [bs,dt,hidden_size]
                print(f"第{i + 1}次的解码器输入:{cur_token_ids}")
                token_embed = self.embed_layer(cur_token_ids)

                # 3. 调用rnn结构获取序列特征向量
                # output: [bs,dt,hidden_size] 每个token对应的特征向量
                # hcn: 当前时刻的状态信息
                output, hcn = self.rnn_layer(token_embed, hcn)

                # 4. 获取最后一个时刻的提取特征向量值
                output_t = output[:, -1, :]

                # 5. 预测属于各个类别的置信度
                score = self.classify_layer(output_t)  # [bs, vocab_size]

                # 6. 获取预测类别id
                pred_ids = torch.argmax(score, dim=1, keepdim=True)  # [bs,1] 获取置信度最大的下标作为预测类别id

                # 7. 将当前时刻的预测结果和之前的结果合并到一起
                token_ids = torch.cat([token_ids, pred_ids], dim=1)  # [bs,dt+1]
                cur_token_ids = pred_ids  # 当前时刻的预测id作为下一个时刻的输入
                i += 1

                # 8. 判断是否结束生成逻辑：一般情况下至少两个条件 预测到结尾token，预测总序列长度超过一定的限制
                if token_ids.shape[1] > 10:
                    break
            return token_ids  # 预测类别id


class Seq2SeqModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, decoder_vocab_size=None, decoder_num_layers=None):
        super().__init__()

        if decoder_vocab_size is None:
            decoder_vocab_size = vocab_size
        if decoder_num_layers is None:
            decoder_num_layers = num_layers
        # 编码器创建
        self.encoder = EncoderModule(vocab_size, hidden_size, num_layers)
        # 解码器创建
        self.decoder = DecoderModule(decoder_vocab_size, hidden_size, decoder_num_layers)

    def forward(self, encoder_token_ids, decoder_token_ids):
        # 1. 获取编码器提取出来的特征向量信息
        encoder_ctx = self.encoder(encoder_token_ids)

        # 2. 获取解码器前向执行信息
        decoder_outputs = self.decoder(decoder_token_ids, encoder_ctx)

        return decoder_outputs


def training():
    seq2seq = Seq2SeqModule(10000, 64, 3, decoder_num_layers=2)
    print(seq2seq)

    loss_fn = nn.CrossEntropyLoss(reduction='none')
    # loss_fn = nn.CrossEntropyLoss()

    # 训练过程测试
    encoder_token_ids = torch.tensor([[12, 35, 26, 34, 253]])  # [1,5]
    decoder_token_ids = torch.tensor([[3, 102, 235, 1523, 2132, 1243]])  # [1,6]
    decoder_target_ids = torch.tensor([[102, 235, 1523, 2132, 1243, 4]])  # [1,6]

    seq2seq.train()
    decoder_score = seq2seq(encoder_token_ids, decoder_token_ids)  # [1,6,10000]
    print(decoder_score.shape)

    loss = loss_fn(torch.permute(decoder_score, dims=(0, 2, 1)), decoder_target_ids)
    print(loss)


def interface():
    # Seq2Seq案例

    seq2seq = Seq2SeqModule(10000, 64, 3, decoder_num_layers=2)
    print(seq2seq)

    # 推理预测过程测试

    encoder_token_ids = torch.tensor([[12, 35, 26, 34, 253]])
    # decoder_token_ids = torch.tensor([[3]])
    decoder_token_ids = torch.tensor([[3, 8026]])

    seq2seq.eval()
    pred_token_ids = seq2seq(encoder_token_ids, decoder_token_ids)

    print(pred_token_ids.shape)
    print(f"预测token id:\n\t{pred_token_ids}")
    # 预测token id:
    # 	tensor([[   3, 8036, 8120, 5712, 5712, 5712, 2137, 2137, 2137, 8036, 8036]])


if __name__ == '__main__':
    # training()
    interface()

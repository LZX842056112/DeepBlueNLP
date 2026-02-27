# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/20 11:36
Create User : 19410
Desc : Attention结构
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 固定随机种子，保证每次运行结果一致，方便调试
torch.manual_seed(24)


# 1. 编码器 (Encoder)
# 作用：将源语言句子压缩成特征向量序列和全局上下文向量
class EncoderModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()

        # 词嵌入层：将单词的ID转为稠密向量表示
        self.embed_layer = nn.Embedding(
            num_embeddings=vocab_size,  # 词汇表大小 --> token到id转换的映射表大小
            embedding_dim=hidden_size  # 每个词/Token对应的特征向量维度大小
        )
        # LSTM层：提取序列特征
        self.rnn_layer = nn.LSTM(
            input_size=hidden_size,  # 每个token输入的特征向量维度大小
            hidden_size=hidden_size,  # 每个token输出的特征向量维度大小
            num_layers=num_layers,  # LSTM层数
            batch_first=True,  # 输入数据格式为 [batch_size, seq_len, feature]
            bidirectional=False  # 单向LSTM (如果是True，输出维度会翻倍)
        )
        # 将RNN输出特征值转换为高阶特征向量C
        # 上下文特征转换层：将LSTM最后的隐藏状态映射为全局上下文向量
        self.ctx_feature_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, token_ids):
        # 1. token id转换为token embedding向量 [bs,et] -> [bs,et,hidden_size]
        token_embed = self.embed_layer(token_ids)
        # 2. 调用rnn结构获取序列特征向量
        # output : 包含所有时间步的隐藏状态 [bs,et,2*hidden_size] --> 当前是双向结构
        # state: 是最后一个时间步的状态 rnn和gru的时候，只有一个值；lstm的时候，有两个值(二元组)；shape均为[?,bs,hidden_size]
        output, state = self.rnn_layer(token_embed)
        # 将 h_n 和 c_n 相加融合 (这是一种处理方式，也可以只取 h_n)
        if isinstance(state, tuple):
            state = state[0] + state[1]
        # 对 num_layers 维度求均值，将多层状态压缩为单层状态 -> 形状: [batch_size, hidden_size]
        state = torch.mean(state, dim=0)  # [?,bs,hidden_size] -->  [bs,hidden_size]
        # 3. 通过线性层+ReLU生成最终的全局上下文向量 [bs,hidden_size] -> [bs,hidden_size]
        ctx_embed = self.ctx_feature_layer(state)

        # output: Encoder所有时刻的输出 (后续作为 Attention 的 Key 和 Value)
        # ctx_embed: 全局上下文向量 (后续用于初始化 Decoder 的隐藏状态)
        return output, ctx_embed  # [bs,t,e] [bs,e]


# 2. 注意力机制 (Attention)
# 作用：计算 Decoder 当前状态与 Encoder 所有输出之间的关联权重
def qkv_attention_value(q, k, v):
    """
    QKV Attention计算
    :param q:  [bs, qt, e] bs个样本，每个样本qt个query，每个query是一个e维的向量
    :param k:  [bs, kvt, e] bs个样本，每个样本kvt个key，每个key是一个e维的向量
    :param v:  [bs, kvt, e] bs个样本，每个样本kvt个value，每个value是一个e维的向量
    :return: [bs, qt, e]
    """
    # Attention计算
    # 1. 计算q和k之间的相关性
    bs, qt, e = q.shape
    # q: [bs, qt, e], k.T: [bs, e, kvt]
    # torch.matmul 矩阵乘法后得到 score 形状: [bs, qt, kvt]
    # (即每个 Query 与所有 Key 的相似度打分)
    # bs个样本，每个样本有qt的query，每个query和kvt个key之间的相关性
    score = torch.matmul(q, torch.transpose(k, dim0=2, dim1=1))
    # 缩放操作：除以维度的平方根，防止维度过大导致内积结果过大，进而导致 softmax 梯度消失 (PPT红字部分)
    score = score / np.sqrt(e)  # [bs, qt, kvt]

    # 2. Softmax 归一化，将打分转化为概率分布 alpha (即注意力权重)
    # alpha 形状: [bs, qt, kvt]
    alpha = torch.softmax(score, dim=-1)  # [bs, qt, kvt]

    # 3. 使用注意力权重对 Value 进行加权求和 [bs, qt, kvt] * [bs, kvt, e] ---> [bs, qt, e]
    value = torch.matmul(alpha, v)

    return value


# Attention 计算的包装函数，适配 Seq2Seq 结构
def attention_value(encoder_output_value, decoder_state):
    """
    计算Attention
    :param encoder_output_value: [bs,t,e] bs个样本，每个样本t个向量
    :param decoder_state: [bs,e] 兼容LSTM/RNN/GRU + 多层的结构
    :return: [bs,e] 针对每个样本，将t个向量合并成一个向量
    """
    # 求均值
    # return torch.mean(encoder_output_value, dim=1)

    # 预处理 Decoder 的状态 (将多层的 h 和 c 压缩成单层) ---> 将解码器状态对象的shape转换为[bs,e]
    if isinstance(decoder_state, tuple):
        decoder_state = decoder_state[0] + decoder_state[1]
    if decoder_state.ndim == 3:
        decoder_state = torch.mean(decoder_state, dim=0)

    # # Attention计算
    # # 1. 计算解码器状态和编码器输出之间的相关性
    # bs, e = decoder_state.shape
    # # [bs,1,e] * [bs,e,t] --> [bs,1,t] bs个样本，每个样本有1的解码器状态，每个解码器状态和t个编码器输出之间的相关性
    # score = torch.matmul(decoder_state[:, None, :], torch.transpose(encoder_output_value, dim0=2, dim1=1))
    # score = score / np.sqrt(e)  # [bs,1,t]
    #
    # # 2. 将相关性转换为权重概率
    # alpha = torch.softmax(score, dim=-1)  # [bs,1,t]
    #
    # # 3. 加权合并 [bs,1,t] * [bs,t,e] ---> [bs,1,e]
    # value = torch.matmul(alpha, encoder_output_value)
    # 结果形状是 [bs, 1, e]，取第 0 个时间步返回 [bs, e]
    # return value[:, 0, :]

    # 将 Decoder 当前状态作为 Query (Q)
    # 将 Encoder 所有时刻输出作为 Key (K) 和 Value (V)
    # decoder_state[:, None, :] 的作用是在中间增加一个维度，从 [bs, e] 变成 [bs, 1, e]
    # 结果形状是 [bs, 1, e]，取第 0 个时间步返回 [bs, e]
    return qkv_attention_value(
        q=decoder_state[:, None, :],
        k=encoder_output_value,
        v=encoder_output_value
    )[:, 0, :]


# 3. 解码器 (Decoder)
# 作用：结合 Attention 机制，逐步生成目标序列
class DecoderModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # 词嵌入层
        self.embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        # 初始化 RNN 隐藏状态的线性层：将 Encoder 的单层 ctx 映射为 Decoder 需要的 num_layers 层状态
        self.rnn_init_h0_layers = nn.Linear(hidden_size, num_layers * hidden_size)
        self.rnn_init_c0_layers = nn.Linear(hidden_size, num_layers * hidden_size)
        # 解码器 LSTM
        self.rnn_layer = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        # 分类器：将隐藏状态映射回词汇表大小，用于预测下一个词的概率
        self.classify_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids, encoder_output_value, encoder_ctx):
        """
            前向执行过程
            : token_ids : [bs,dt] 解码器输入token ids
            : encoder_output_value: [bs,t,hidden_size] 编码器输出的各个时刻的特征向量
            : encoder_ctx : [bs, hidden_size] 编码器提取出来的文本特征向量
        """
        # 1. 针对编码器提取特征向量进行特征提取，获取lstm的初始状态
        # 1. 初始化 Decoder 的初始状态 (h0, c0)
        bs, e = encoder_ctx.shape
        # 将 ctx [bs, e] -> [bs, num_layers * e] -> [bs, e, num_layers] -> [num_layers, bs, e]
        # 这是为了满足 LSTM 对 h0/c0 的形状要求：(num_layers, batch_size, hidden_size)
        h0 = self.rnn_init_h0_layers(encoder_ctx)  # [bs,hidden_size] -> [bs,num_layers*hidden_size]
        h0 = h0.reshape((bs, e, -1))  # [bs,num_layers*hidden_size] --> [bs,hidden_size,num_layers]
        h0 = torch.permute(h0, dims=(2, 0, 1))  # [bs,hidden_size,num_layers] -> [num_layers,bs,hidden_size]

        c0 = self.rnn_init_c0_layers(encoder_ctx)  # [bs,hidden_size] -> [bs,num_layers*hidden_size]
        c0 = c0.reshape((bs, e, -1))  # [bs,num_layers*hidden_size] --> [bs,hidden_size,num_layers]
        c0 = torch.permute(c0, dims=(2, 0, 1))  # [bs,hidden_size,num_layers] -> [num_layers,bs,hidden_size]

        # 训练模式 (Teacher Forcing 模式)
        if self.training:
            # 2. 针对输入数据进行embedding操作 [bs,dt] -> [bs,dt,hidden_size]
            token_embed = self.embed_layer(token_ids)

            bs, dt, e = token_embed.shape
            # 初始化状态
            hcn = (h0, c0)
            scores = []
            # 【核心】：在时间步上循环 (因为 Attention 依赖前一步的状态，所以不能像普通 RNN 一次性塞进去)
            for _t in range(dt):
                # 将编码器特征向量和解码器的token向量进行合并
                atte_value = attention_value(encoder_output_value, hcn)
                atte_value = atte_value[:, None]  # [bs,e] --> [bs,1,e]
                # [bs,1,hidden_size] +  [bs,1,hidden_size]
                # 【融合 Attention】：将当前词的 Embedding 与 Attention 向量相加 (也有做法是拼接 concat)
                cur_token_embed = token_embed[:, _t:_t + 1, :] + atte_value

                # 3. 调用rnn结构获取序列特征向量
                # output: [bs,1,hidden_size] 每个token对应的特征向量
                # hcn: 当前时刻的状态信息
                # 丢入 LSTM 执行单步计算
                output, hcn = self.rnn_layer(cur_token_embed, hcn)

                # 4. 针对每个token进行全连接得到预测对应类别 [bs,1,hidden_size] -> [bs,1,vocab_size]
                score = self.classify_layer(output)
                scores.append(score)
            # 将所有时间步的预测结果拼接起来 -> [bs, dt, vocab_size]
            return torch.concat(scores, dim=1)
        # 推理预测模式 (自回归生成)
        else:
            ##### 针对推理预测过程，需要一个token、一个token进行输入预测得到结果
            i = 0
            # 通常是以特殊的 <BOS> (句首标志) token 开始
            cur_token_ids = token_ids  # 当前时刻的token id
            hcn = (h0, c0)

            while True:
                # 2. 针对输入数据进行embedding操作 [bs,dt] -> [bs,dt,hidden_size]
                print(f"第{i + 1}次的解码器输入:{cur_token_ids}")
                # 获取输入序列的 Embedding
                token_embed = self.embed_layer(cur_token_ids)

                # 将编码器特征向量和解码器的token向量进行合并
                # 计算 Attention
                atte_value = attention_value(encoder_output_value, hcn)
                atte_value = atte_value[:, None]  # [bs,e] --> [bs,1,e]
                # 将编码器特征向量和解码器的token向量进行合并
                # [bs,dt,hidden_size] +  [bs,1,hidden_size]
                token_embed = token_embed + atte_value

                # 3. 调用rnn结构获取序列特征向量
                # output: [bs,dt,hidden_size] 每个token对应的特征向量
                # hcn: 当前时刻的状态信息
                output, hcn = self.rnn_layer(token_embed, hcn)

                # 4. 获取最后一个时刻的提取特征向量值
                output_t = output[:, -1, :]

                # 5. 预测属于各个类别的置信度
                score = self.classify_layer(output_t)  # [bs, vocab_size]

                # 6. 获取置信度最大的类别 ID (贪婪解码 Greedy Decoding)
                pred_ids = torch.argmax(score, dim=1, keepdim=True)  # [bs,1] 获取置信度最大的下标作为预测类别id

                # 7. 将当前时刻的预测结果和之前的结果合并到一起
                token_ids = torch.cat([token_ids, pred_ids], dim=1)  # [bs,dt+1]
                cur_token_ids = pred_ids  # 当前时刻的预测id作为下一个时刻的输入
                i += 1

                # 8. 判断是否结束生成逻辑：一般情况下至少两个条件 预测到结尾token，预测总序列长度超过一定的限制
                # 停止条件：生成长度超过限制，或者生成了 <EOS> 结束符(此处代码省略了EOS判断，仅保留长度截断)
                if token_ids.shape[1] > 10:
                    break
            return token_ids  # 预测类别id


# 4. Seq2Seq 整体封装模块
# 作用：将 Encoder 和 Decoder 连接在一起，形成完整的黑盒模型
class Seq2SeqModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, decoder_vocab_size=None, decoder_num_layers=None):
        super().__init__()
        # 如果没有单独指定 Decoder 的词表大小和层数，则默认与 Encoder 保持一致
        # (例如机器翻译中，中英文词表大小通常不同，所以这里提供了灵活性)
        if decoder_vocab_size is None:
            decoder_vocab_size = vocab_size
        if decoder_num_layers is None:
            decoder_num_layers = num_layers
        # 编码器创建
        self.encoder = EncoderModule(vocab_size, hidden_size, num_layers)
        # 解码器创建
        self.decoder = DecoderModule(decoder_vocab_size, hidden_size, decoder_num_layers)

    def forward(self, encoder_token_ids, decoder_token_ids):
        # 1. 编码阶段：将源句子输入 Encoder，获取所有时刻的输出(用于Attention)和全局上下文(用于初始化)
        encoder_output_values, encoder_ctx = self.encoder(encoder_token_ids)
        # 2. 解码阶段：将目标句子的前缀、Encoder的输出、Encoder的上下文一并送入 Decoder
        decoder_outputs = self.decoder(decoder_token_ids, encoder_output_values, encoder_ctx)
        # 返回 Decoder 的预测结果 (训练时返回每个词的概率分布，推理时返回预测的词ID列表)
        return decoder_outputs


# 5. 模拟训练过程
def training():
    # 初始化一个 Seq2Seq 模型
    # 词表大小 10000，隐藏层维度 64，Encoder 3层，Decoder 2层
    seq2seq = Seq2SeqModule(10000, 64, 3, decoder_num_layers=2)
    print(seq2seq)

    # 定义交叉熵损失函数
    # reduction='none' 表示不对每个 batch 的 loss 求平均，而是返回每个样本/每个词的 loss，方便后续按需做 Mask 处理
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    # loss_fn = nn.CrossEntropyLoss()

    # ---------------- 准备伪造的训练数据 ----------------
    # 假设：源语言句子 (比如中文 "我 爱 你")
    # 训练过程测试
    encoder_token_ids = torch.tensor([[12, 35, 26, 34, 253]])  # [1,5]
    # 假设：目标语言句子的输入 (比如英文 "<BOS> I love you")
    # 注意：这里的 3 通常代表起始符 <BOS> 或 <SOS>
    decoder_token_ids = torch.tensor([[3, 102, 235, 1523, 2132, 1243]])  # [1,6]
    # 假设：目标语言句子的真实标签 (比如英文 "I love you <EOS>")
    # 注意：标签比输入往后错开了一位！最后的 4 通常代表结束符 <EOS>
    decoder_target_ids = torch.tensor([[102, 235, 1523, 2132, 1243, 4]])  # [1,6]

    # 开启训练模式 (这会激活 Decoder 内部的 Teacher Forcing 逻辑)
    seq2seq.train()
    # 前向传播，获取预测分数
    decoder_score = seq2seq(encoder_token_ids, decoder_token_ids)  # [1,6,10000]
    # 输出形状为: [batch_size=1, seq_len=6, vocab_size=10000]
    print(decoder_score.shape)

    # ---------------- 计算 Loss ----------------
    # PyTorch 的 CrossEntropyLoss 针对多维序列数据，要求把“类别维度(vocab_size)”放在第二维
    # 即要求模型输出的形状是 [batch_size, num_classes, seq_len]
    # 所以需要用 torch.permute 将 [1, 6, 10000] 维度转换成 [1, 10000, 6]
    loss = loss_fn(torch.permute(decoder_score, dims=(0, 2, 1)), decoder_target_ids)
    print(loss)


# 6. 模拟推理(预测)过程
def interface():
    # Seq2Seq案例
    seq2seq = Seq2SeqModule(10000, 64, 3, decoder_num_layers=2)
    print(seq2seq)

    # 推理预测过程测试
    encoder_token_ids = torch.tensor([[12, 35, 26, 34, 253]])
    # 推理时，Decoder 一开始只有起始符 <BOS> (代表ID: 3)
    # 或者给一个特定的前缀，让它接着往下续写 (比如给了 [3, 8026])
    # decoder_token_ids = torch.tensor([[3]])
    decoder_token_ids = torch.tensor([[3, 8026]])

    # 开启评估模式 (这会激活 Decoder 内部的 while True 自回归生成逻辑)
    seq2seq.eval()
    # 执行预测 (此时不需要传 target)
    pred_token_ids = seq2seq(encoder_token_ids, decoder_token_ids)

    print(pred_token_ids.shape)
    print(f"预测token id:\n\t{pred_token_ids}")


if __name__ == '__main__':
    training()
    interface()

# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/20 11:36
Create User : 19410
Desc : 最原始的结构：解码器重复执行每一次的操作
       (注：指在推理阶段，每次生成新词时，都会把从开头到当前的所有词重新输入一遍LSTM计算)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置固定的随机种子，确保每次运行代码时初始化的权重相同，方便复现结果
torch.manual_seed(24)


# ==========================================
# 1. 编码器模块 (Encoder)
# 作用：将源文本序列压缩成一个固定维度的上下文向量
# ==========================================
class EncoderModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()

        # 词嵌入层：将离散的 token ID 转换为稠密的向量表示
        self.embed_layer = nn.Embedding(
            num_embeddings=vocab_size,  # 词汇表大小
            embedding_dim=hidden_size  # 词向量维度大小 (隐藏层维度)
        )

        # 编码器核心：双向 LSTM
        self.rnn_layer = nn.LSTM(
            input_size=hidden_size,  # 输入特征维度
            hidden_size=hidden_size,  # 隐藏状态特征维度
            num_layers=num_layers,  # LSTM 层数
            batch_first=True,  # 输入 shape 为 [batch_size, seq_len, feature]
            bidirectional=True  # 使用双向 LSTM 捕获上下文信息
        )

        # 特征提取层：由于是双向LSTM，状态维度会翻倍或变得复杂，这里用全连接层将其统一映射为上下文向量 C
        self.ctx_feature_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, token_ids):
        # 1. 词嵌入映射: [bs, et] -> [bs, et, hidden_size]  (bs=batch_size, et=encoder_token_len)
        token_embed = self.embed_layer(token_ids)

        # 2. LSTM 序列计算
        # output: 所有时间步的顶层隐藏状态。因为是双向，shape 为 [bs, et, 2 * hidden_size]
        # state: 最后一个时间步的隐藏状态和细胞状态。对于 LSTM 是一个元组 (h_n, c_n)
        output, state = self.rnn_layer(token_embed)

        # 处理最终状态
        if isinstance(state, tuple):
            # 将隐状态 h_n 和 细胞状态 c_n 简单相加进行特征融合
            state = state[0] + state[1]

            # 此时 state 包含多层和双向的状态 [num_layers * 2, bs, hidden_size]
        # 沿着第 0 维度取均值，将其压缩为单层的状态表示 -> [bs, hidden_size]
        state = torch.mean(state, dim=0)

        # 3. 经过全连接层得到最终的上下文特征向量 ctx_embed: [bs, hidden_size]
        ctx_embed = self.ctx_feature_layer(state)

        return ctx_embed


# ==========================================
# 2. 解码器模块 (Decoder)
# 作用：根据编码器生成的上下文向量，逐步生成目标文本序列
# ==========================================
class DecoderModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.num_layers = num_layers

        # 解码器的词嵌入层
        self.embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)

        # 初始化状态映射层：将编码器传来的单层上下文向量，映射为解码器 LSTM 需要的多层初始状态
        self.rnn_init_h0_layers = nn.Linear(hidden_size, num_layers * hidden_size)
        self.rnn_init_c0_layers = nn.Linear(hidden_size, num_layers * hidden_size)

        # 解码器核心：单向 LSTM (生成任务只能基于过去预测未来，不能用双向)
        self.rnn_layer = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        # 输出分类层：将 LSTM 输出的特征映射回词表大小，用于预测下一个词的概率
        self.classify_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids, encoder_ctx):
        """
            前向执行过程
            : token_ids : [bs, dt] 解码器输入token ids (dt=decoder_token_len)
            : encoder_ctx : [bs, hidden_size] 编码器提取出来的文本特征向量
        """
        # 1. 准备解码器 LSTM 的初始隐状态 (h0) 和细胞状态 (c0)
        bs, e = encoder_ctx.shape

        # 计算 h0: [bs, hidden_size] -> [bs, num_layers * hidden_size] -> reshape -> permute
        # 最终 h0 形状必须满足 LSTM 格式: [num_layers, bs, hidden_size]
        h0 = self.rnn_init_h0_layers(encoder_ctx)
        h0 = h0.reshape((bs, e, -1))
        h0 = torch.permute(h0, dims=(2, 0, 1))

        # 计算 c0: 过程同上。最终 c0 形状: [num_layers, bs, hidden_size]
        c0 = self.rnn_init_c0_layers(encoder_ctx)
        c0 = c0.reshape((bs, e, -1))
        c0 = torch.permute(c0, dims=(2, 0, 1))

        # ---------------- 训练模式：并行计算 (Teacher Forcing) ----------------
        if self.training:
            # 2. 输入词嵌入 [bs, dt] -> [bs, dt, hidden_size]
            token_embed = self.embed_layer(token_ids)

            # 3. 输入 LSTM。将之前算好的 h0, c0 作为初始状态传入。
            # 训练时，所有真实的目标 token 都会一次性输入，output 包含每个时间步的输出特征。
            output, _ = self.rnn_layer(token_embed, (h0, c0))

            # 4. 预测每个时间步的下一个词类别得分 [bs, dt, hidden_size] -> [bs, dt, vocab_size]
            score = self.classify_layer(output)

            return score  # 返回预测得分用于计算 Loss

        # ---------------- 推理模式：自回归循环生成 (最原始的结构) ----------------
        else:
            i = 0
            while True:  # 死循环，直到达到终止条件
                # 打印当前输入序列，展示“最原始的结构”的特点：每次序列长度都在增加
                print(f"第{i + 1}次的解码器输入:{token_ids}")

                # 2. 将截止当前生成的所有 token 进行 embedding: [bs, dt] -> [bs, dt, hidden_size]
                token_embed = self.embed_layer(token_ids)

                # 3. 核心冗余点：每次生成一个新词，都要把从头到尾的整个序列重新放进 LSTM 计算一遍！
                # 且每次都使用最初始的 (h0, c0)。虽然能算出正确结果，但前面词的计算被重复了无数次。
                output, _ = self.rnn_layer(token_embed, (h0, c0))

                # 4. 我们只需要最后一个时间步的输出特征来预测下一个词
                output_t = output[:, -1, :]  # 切片: [bs, hidden_size]

                # 5. 计算最终时刻属于各个词汇类别的置信度/得分: [bs, vocab_size]
                score = self.classify_layer(output_t)

                # 6. 获取得分最高的索引，即预测出的新词的 ID: [bs, 1]
                pred_ids = torch.argmax(score, dim=1, keepdim=True)

                # 7. 将新预测出来的词 ID 拼接到现有序列末尾，作为下一轮循环的输入: 长度 dt 变为 dt+1
                token_ids = torch.cat([token_ids, pred_ids], dim=1)
                i += 1

                # 8. 终止条件：生成的序列长度大于 10 时强制停止。
                # （实际业务中通常还会加入判断 pred_ids 是否为特定的 <EOS> 结束符）
                if token_ids.shape[1] > 10:
                    break

            return token_ids  # 返回最终生成的一段完整的序列 ID


# ==========================================
# 3. 整体模型组装
# ==========================================
class Seq2SeqModule(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, decoder_vocab_size=None, decoder_num_layers=None):
        super().__init__()

        # 如果解码器的词表大小或层数未指定，则默认与编码器相同
        if decoder_vocab_size is None:
            decoder_vocab_size = vocab_size
        if decoder_num_layers is None:
            decoder_num_layers = num_layers

        # 实例化编码器和解码器
        self.encoder = EncoderModule(vocab_size, hidden_size, num_layers)
        self.decoder = DecoderModule(decoder_vocab_size, hidden_size, decoder_num_layers)

    def forward(self, encoder_token_ids, decoder_token_ids):
        # 1. 编码器提取源文本的上下文特征
        encoder_ctx = self.encoder(encoder_token_ids)

        # 2. 解码器根据上下文特征和目标输入前向执行
        decoder_outputs = self.decoder(decoder_token_ids, encoder_ctx)

        return decoder_outputs


# ==========================================
# 测试函数 1: 训练过程
# ==========================================
def training():
    print("--- 训练过程测试 ---")
    # 初始化模型: 词表大小 10000, 隐藏层 64, encoder 3 层, decoder 2 层
    seq2seq = Seq2SeqModule(10000, 64, 3, decoder_num_layers=2)

    # 交叉熵损失函数，不取平均 (便于查看每个位置的 loss)
    loss_fn = nn.CrossEntropyLoss(reduction='none')

    # 模拟输入数据:
    # 编码器输入 [1, 5] (比如: "我 是 中 国 人")
    encoder_token_ids = torch.tensor([[12, 35, 26, 34, 253]])
    # 解码器输入 [1, 6] (以起始符 id=3 开头，比如: "<bos> I am a Chi nese")
    decoder_token_ids = torch.tensor([[3, 102, 235, 1523, 2132, 1243]])
    # 解码器目标预测 [1, 6] (对应错位一个 token，以结束符 id=4 结尾，比如: "I am a Chi nese <eos>")
    decoder_target_ids = torch.tensor([[102, 235, 1523, 2132, 1243, 4]])

    # 开启训练模式
    seq2seq.train()
    # 前向传播得到预测分数/logits: shape -> [batch_size, seq_len, vocab_size] = [1, 6, 10000]
    decoder_score = seq2seq(encoder_token_ids, decoder_token_ids)
    print(f"训练阶段 Decoder 输出形状: {decoder_score.shape}")

    # 计算损失：CrossEntropyLoss 需要 input 的形状是 [batch_size, num_classes, ...]
    # 因此使用 torch.permute 将维度调换为 [1, 10000, 6]
    loss = loss_fn(torch.permute(decoder_score, dims=(0, 2, 1)), decoder_target_ids)
    print(f"训练阶段 Loss: {loss}\n")


# ==========================================
# 测试函数 2: 推理(生成)过程
# ==========================================
def interface():
    print("--- 推理预测过程测试 ---")
    seq2seq = Seq2SeqModule(10000, 64, 3, decoder_num_layers=2)

    # 模拟源文本输入
    encoder_token_ids = torch.tensor([[12, 35, 26, 34, 253]])
    # 推理时，初始只需要给解码器喂入一个起始符 (BOS = Begin Of Sequence, id=3)
    decoder_token_ids = torch.tensor([[3]])

    # 开启评估(推理)模式，触发解码器代码中的 else 分支
    seq2seq.eval()
    # 开始自回归生成，直到长度达到限制
    pred_token_ids = seq2seq(encoder_token_ids, decoder_token_ids)

    print(f"最终预测输出形状: {pred_token_ids.shape}")
    print(f"预测token id:\n\t{pred_token_ids}")


if __name__ == '__main__':
    training()
    interface()
# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/21 16:21
Create User : 19410
Desc : Bert模型的使用Demo

NOTE:
    默认情况下，会下载到当前用户根目录下的.cache/huggingface文件夹中
    但是可以通过给定环境变量:
        XDG_CACHE_HOME=xxx 来指定模型保存文件路径

# pip install transformers==4.57.3 -i https://mirrors.aliyun.com/pypi/simple
# 由于transformers框架对应的网站 https://huggingface.co 需要外网访问，所以这里弄一个国内使用的网站
# 国内网站：https://hf-mirror.com   --- 有做访问限制的处理

"""

import os

import torch

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 开启国内镜像加速下载
os.environ['XDG_CACHE_HOME'] = r"D:\cache"  # 更改缓存路径到 D 盘
os.environ['CACHE_HOME'] = r'D:\cache'


# tt_v0: 基础 API 演示 - 加载预训练分词器和模型
def tt_v0():
    from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, BertModel

    # 从 HuggingFace 加载指定的中文预训练分词器 (自动下载 vocab.txt 等)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    print(f"Bert对应的分词器:\n{type(tokenizer)}\n{tokenizer}\n")
    # 加载预训练模型权重 (自动下载 pytorch_model.bin)
    model = BertModel.from_pretrained("bert-base-chinese")
    print(f"Bert对应的模型:\n{type(model)}\n{model}\n")


# tt_v1: 高级 API 演示 - 使用 Pipeline 进行掩码预测任务 (MLM)
def tt_v1():
    from transformers import pipeline

    # 创建一个 'fill-mask' (完形填空) 的流水线，并指定使用 bert-base-chinese
    unmasker = pipeline('fill-mask', model='bert-base-chinese')
    print("=" * 50)
    # 预测英文句子中的 [MASK] (BERT 在训练时见过中英混合语料)
    print(unmasker("The man worked as a [MASK]."))
    print("=" * 50)
    print(unmasker("The woman worked as a [MASK]."))
    print("=" * 50)
    # 预测中文句子中的被遮盖字
    print(unmasker("中国的首都是[MASK]京。"))


# tt_v2: 核心用法 - 文本特征提取与 [CLS] 向量获取
def tt_v2():
    from transformers import BertTokenizer, BertModel, BertConfig
    from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained("bert-base-chinese")
    text = "还有双鸭山到淮阴的汽车票吗13号的"

    # 基于给定文本得到文本对应的分词结果以及token id转换结果
    # encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input = tokenizer([text, "从这里怎么回家"], return_tensors='pt', padding=True)
    # 将token id输入模型得到模型输出：默认仅包括bert最后一层的输出特征向量
    output = model(**encoded_input)
    print(type(output))

    # 获取bert模型最后一层的输出特征向量 [bs,t,hidden_size]
    last_hidden_state = output[0]  # output.last_hidden_state
    print(last_hidden_state.shape)

    # 获取每一句话的第一个 Token ([CLS]) 对应的特征向量，常用于文本分类
    cls_hidden_state = last_hidden_state[:, 0, :]
    print(cls_hidden_state.shape)


# tt_v3: 进阶用法 - 从零自定义 BERT 配置 (随机初始化参数)
def tt_v3():
    from transformers import BertConfig, BertModel

    # 1. 不加载预训练权重，而是手动定义模型结构的各个超参数
    config = BertConfig(
        vocab_size=1000,  # 词表大小设为 1000
        hidden_size=128,  # 隐藏层特征维度设为 128 (标准bert是768)
        num_hidden_layers=2,  # 只堆叠 2 层 Transformer Encoder (标准bert是12)
        num_attention_heads=4,  # 多头注意力的头数为 4
        intermediate_size=512,  # FFN 前馈神经网络的中间维度大小
        hidden_act="gelu",  # 激活函数
        hidden_dropout_prob=0.1,  # 神经元随机失活率
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,  # 最大支持句子长度为 512
        type_vocab_size=2,  # 句子类型 (Segment ID) 种类为 2
        initializer_range=0.02,  # 参数初始化的标准差
        layer_norm_eps=1e-12,  # LayerNorm 的极小值防止除以0
        pad_token_id=0,  # 填充字符的 ID
        position_embedding_type="absolute",  # 使用绝对位置编码
        use_cache=True,
        classifier_dropout=None,
    )
    # 基于配置实例化模型 (这是一个全新的、未经训练的模型)
    # add_pooling_layer=False 表示不要池化层 (即不额外对 [CLS] 专门做一个全连接层)
    model = BertModel(config=config, add_pooling_layer=False)

    # 前向传播：传入自造的 dummy 数据
    bert_output = model(
        # 假设 101 是 [CLS], 102 是 [SEP], 0 是 [PAD]
        input_ids=torch.tensor([
            [101, 25, 36, 12, 102, 0, 0],
            [101, 5, 8, 9, 32, 58, 102]
        ]),
        # 1 表示真实词语(需要进行 Attention 交互)，0 表示填充词(忽略不计)
        attention_mask=torch.tensor([
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1]
        ]),
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=True,  # 开启后，不仅返回特征，还返回每一层的注意力权重矩阵！
        output_hidden_states=True  # 开启后，不仅返回最后一层，还返回每一层的特征向量！
    )
    print(type(bert_output))
    """
    bert_output中的参数: ---> 以dict形式返回的时候
        last_hidden_state: Optional[torch.FloatTensor] = None 最后一层的特征向量
        pooler_output: Optional[torch.FloatTensor] = None CLS对应位置的全连接特征向量
        hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None 所有层的特征向量
        past_key_values: Optional[Cache] = None 需要缓存的key/value信息 -- 解码器生效
        attentions: Optional[tuple[torch.FloatTensor, ...]] = None self attention计算出来的相关性矩阵
        cross_attentions: Optional[tuple[torch.FloatTensor, ...]] = None encoder-decoder cross attention计算出来的相关性矩阵
    """


# tt_v4: 终极进阶 - 充当 Decoder 结构并使用 KV Cache
def tt_v4():
    from transformers import BertConfig, BertModel

    config = BertConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=512,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        is_decoder=True  # 【关键】将模型转变为解码器模式 (如 GPT 一般)，具有因果掩码(Causal Mask)机制 -- 只有解码器，没有编码器的结构
    )
    model = BertModel(config=config, add_pooling_layer=False)

    # 阶段一：处理初始文本 (Prompt)
    bert_output = model(
        input_ids=torch.tensor([
            [101, 25, 36, 12, 102, 0, 0],
            [101, 5, 8, 9, 32, 58, 102]
        ]),
        attention_mask=torch.tensor([
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1]
        ]),
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=True,  # 【关键】开启缓存，让模型记住当前计算出来的 Key 和 Value
        output_attentions=True,  # 期望返回attention的权重列表
        output_hidden_states=True  # 期望返回各个层的输出特征向量
    )
    print(type(bert_output))
    """
    bert_output中的参数: ---> 以dict形式返回的时候
        last_hidden_state: Optional[torch.FloatTensor] = None 最后一层的特征向量
        pooler_output: Optional[torch.FloatTensor] = None CLS对应位置的全连接特征向量
        hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None 所有层的特征向量
        past_key_values: Optional[Cache] = None 需要缓存的key/value信息 -- 解码器生效
        attentions: Optional[tuple[torch.FloatTensor, ...]] = None self attention计算出来的相关性矩阵
        cross_attentions: Optional[tuple[torch.FloatTensor, ...]] = None encoder-decoder cross attention计算出来的相关性矩阵
    """

    # 阶段二：文本生成中的下一步预测
    # 注意看：这里输入序列长度只有 1 了！(模拟模型刚才预测出了一个新词，现在要把新词丢进去继续预测)
    bert_output = model(
        input_ids=torch.tensor([
            [0],  # 第一句话生成了 0
            [135]  # 第二句话生成了 135
        ]),
        attention_mask=torch.tensor([
            [0],
            [1]
        ]),
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        # 【极其关键】将上次计算保留下来的 past_key_values 传回给模型
        past_key_values=bert_output.past_key_values,  # 将上一次输出的缓存key/value再一次输入
        use_cache=True,
        output_attentions=True,  # 期望返回attention的权重列表
        output_hidden_states=True  # 期望返回各个层的输出特征向量
    )
    print(type(bert_output))


if __name__ == '__main__':
    # tt_v0()
    # tt_v1()
    tt_v2()
    # tt_v3()
    # tt_v4()

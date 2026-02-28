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

# 设置 Hugging Face 的国内镜像源，解决下载慢/断连的问题（此处被注释，需要时可打开）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 修改模型默认的缓存下载路径到 D 盘，防止挤爆系统盘
os.environ['XDG_CACHE_HOME'] = r"D:\cache"
os.environ['CACHE_HOME'] = r'D:\cache'


# 测试函数 0: 最基础的模型与分词器加载
def tt_v0():
    # AutoTokenizer 和 AutoModel 是工厂类，能根据传入的模型名称自动实例化对应的类
    from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, BertModel

    # 1. 加载分词器 (Tokenizer)
    # "bert-base-chinese" 是 HuggingFace 上的一个经典中文 BERT 模型。
    # 这一步会下载词表文件 (vocab.txt) 等，用于将中文文本切分成 token 并转为 ID。
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    print(f"Bert对应的分词器:\n{type(tokenizer)}\n{tokenizer}\n")

    # 2. 加载预训练模型 (Model)
    # 这一步会下载模型的权重文件 (pytorch_model.bin，大约 400MB+)。
    model = BertModel.from_pretrained("bert-base-chinese")
    print(f"Bert对应的模型:\n{type(model)}\n{model}\n")


# 测试函数 1: 使用 Pipeline 进行掩码预测 (完形填空)
def tt_v1():
    from transformers import pipeline

    # pipeline 是 transformers 提供的高级 API。
    # 'fill-mask' 任务直接对应了 BERT 预训练时的 MLM (Masked Language Model) 任务。
    unmasker = pipeline('fill-mask', model='bert-base-chinese')

    print("=" * 50)
    # 虽然是中文模型，但它也包含基础的英文字符。预测 "[MASK]" 处的词。
    print(unmasker("The man worked as a [MASK]."))

    print("=" * 50)
    print(unmasker("The woman worked as a [MASK]."))

    print("=" * 50)
    # 测试纯中文的完形填空能力
    print(unmasker("中国的首都是[MASK]京。"))


# 测试函数 2: 底层 API 提取文本特征向量 (最常用的业务场景)
def tt_v2():
    from transformers import BertTokenizer, BertModel, BertConfig
    from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

    # 1. 显式地实例化 BERT 专用的分词器和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained("bert-base-chinese")

    # 定义需要处理的文本
    text = "还有双鸭山到淮阴的汽车票吗13号的"

    # 2. 分词与编码
    # 传入列表表示处理一个 Batch (包含两句话)。
    # return_tensors='pt' 表示返回 PyTorch 的 Tensor 格式。
    # padding=True 表示因为两句话长度不同，自动用 <PAD> (ID为0) 补齐到相同长度。
    encoded_input = tokenizer([text, "从这里怎么回家"], return_tensors='pt', padding=True)

    # 3. 前向传播提取特征
    # **encoded_input 会解包字典，把 input_ids, token_type_ids, attention_mask 等传入模型
    output = model(**encoded_input)
    print(type(output))  # 输出类型是一个包装类

    # 4. 获取特征向量
    # output[0] 等价于 output.last_hidden_state，是模型最后一层的输出
    # 维度形状为: [batch_size, sequence_length, hidden_size]
    last_hidden_state = output[0]
    print(last_hidden_state.shape)  # 例如 [2, 18, 768]，18 是补齐后的最大长度，768 是特征维度

    # 5. 提取句子的全局特征 (CLS Token)
    # BERT 默认会在每个句子最前面强行塞入一个特殊的占位符 [CLS]。
    # 下标 0 代表取序列的第一个 token (即 [CLS]) 的所有特征。
    # 经过了深层 Transformer 的自注意力计算，[CLS] 的向量已经融合了整句话的全局语义信息，通常用于做文本分类任务。
    cls_hidden_state = last_hidden_state[:, 0, :]
    print(cls_hidden_state.shape)  # 维度变为 [2, 768] (bs, hidden_size)


if __name__ == '__main__':
    # tt_v0()
    # tt_v1()
    tt_v2()

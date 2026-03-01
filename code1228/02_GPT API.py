# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/28 11:59
Create User : 19410
Desc : xxx

https://modelscope.cn/models/openai-community/gpt2/files
https://hf-mirror.com/openai-community/gpt2
"""

import os

import torch

# 如果你在国内下载模型很慢，可以取消下面这行的注释，使用国内镜像加速
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ⚠️ 极其重要：更改 Hugging Face 的默认缓存目录！
# 默认情况下，HF 会把动辄几十GB的模型下到 C盘 (C:\Users\用户名\.cache\huggingface)
# 这里将其重定向到 D盘，拯救你的 C盘 空间！
os.environ['XDG_CACHE_HOME'] = r"D:\cache"
os.environ['CACHE_HOME'] = r'D:\cache'


# 方式一：纯粹下载/加载模型 (验证网络和环境)
def download_model():
    # Auto 系列是 HF 最强大的“自动挡”工具。你只要给它名字，它自动帮你找对应的类。
    from transformers import AutoModel, AutoTokenizer

    # 指定要下载的模型 ID (在 HuggingFace 网站上的名字)
    model_id = "openai-community/gpt2"

    # 加载分词器 (把文本变成数字 ID)
    # from_pretrained 会先去刚才设置的 D盘缓存找，找不到就会去网上下载
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 加载模型本体 (加载神经网络权重)
    model = AutoModel.from_pretrained(model_id)

    print(type(tokenizer))  # 输出将会是 GPT2TokenizerFast
    print(type(model))  # 输出将会是 GPT2Model
    print(model)  # 打印模型的网络结构 (你可以看到里面的 Embedding 和多层 Transformer)


# 方式二：“傻瓜式”文本生成 (Pipeline) - 推荐新手使用
def use_gpt_pipeline():
    from transformers import pipeline, set_seed
    from transformers.pipelines.text_generation import TextGenerationPipeline
    # 这里的 path 可以是网上的 model_id，也可以是你本地已经下载好的文件夹路径
    path = r"openai-community/gpt2"
    print(f"模型加载路径:{path}")
    """
    pipeline: 将所有操作合并到一起，达到一个效果：给定一个输入，得到一个明确的输出
        eg: 给定了文本前缀的，得到生成好的完整文本数据
        它把 "文本输入 -> Tokenizer分词 -> Model前向传播 -> 拿概率最高的词 -> 还原成文本" 这套复杂的流程全部封装进了一个黑盒子里。
    """
    pipe = pipeline(
        'text-generation',  # 指定任务类型为“文本生成”
        model=path,  # 指定模型
        framework="pt"  # pt 代表 PyTorch (tf 代表 TensorFlow)
    )
    print(type(pipe))
    print(type(pipe.model))
    print(type(pipe.tokenizer))
    # 固定随机种子，保证每次生成的文本是一样的（方便复现Bug）
    set_seed(42)

    # 喂给模型一句前缀，让它续写！
    # max_length=100: 最多生成 100 个 token
    # num_return_sequences=5: 给我生成 5 种不同的续写结果
    # r = pipe("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
    r = pipe("我是中国人，我比较喜欢吃", max_length=100, num_return_sequences=5)
    print(r)


# 方式三：“硬核”手动控制模型推理 - 适合进阶开发
def use_gpt_model():
    # 不用 Auto，直接导入 GPT2 专属的类
    # 注意：这里用的是 GPT2LMHeadModel (带有语言模型头的版本，专门用来做文本生成的)
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    # 这里你使用的是本地绝对路径，说明你之前已经把模型下到本地了
    path = r"D:\cache\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
    print(f"模型加载路径:{path}")

    # GPT2 中的分词采用的是 BBPE (Byte-Level Byte-Pair Encoding，字节级字节对编码)
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    model = GPT2LMHeadModel.from_pretrained(path)

    text = "我是中国"
    # 1. 分词：把文本变成模型认识的 token IDs
    # return_tensors='pt' 表示直接返回 PyTorch 的 Tensor 格式
    # text = "a b c d e f g h i"
    # 分词+词id转换 [1,6] eg:[[22755,   239, 42468, 40792, 32368,   121]]
    input_ids = tokenizer(text, return_tensors='pt')['input_ids']
    # 2. 前向传播：把 input_ids 喂给模型
    # 获取gpt模型的输出
    gpt_output = model(input_ids)
    # 3. 获取输出：logits 就是模型对字典里每个词打出的“分数” (还未经过 Softmax 的原始分数)
    # 形状为 [batch_size, sequence_length, vocab_size] -> [1, 输入token数, 50257(GPT2的词表大小)]
    # 每个样本的每个token预测属于C个类别的置信度 [bs,t,vocab_size]
    pred_token_scores = gpt_output.logits  # [1,6,50257]
    # 4. 贪心搜索：在最后一个维度 (dim=-1, 即词表维度) 上找分数最高的那个词的 ID
    pred_token_ids = torch.argmax(pred_token_scores, dim=-1)  # [1,6,50257] --> [1,6]

    # 5. 解码：把预测出来的数字 ID 还原成人类能看懂的文本
    text = tokenizer.decode(
        pred_token_ids[0][-1:],  # 我们只关心模型预测的最后一个词 (下一个词)
        skip_special_tokens=True,  # 忽略 <BOS>, <EOS> 等特殊符号
        clean_up_tokenization_spaces=True,  # 清理多余的空格
    )
    print("模型预测的下一个字是:", text)


if __name__ == '__main__':
    # download_model()
    use_gpt_pipeline()
    # use_gpt_model()

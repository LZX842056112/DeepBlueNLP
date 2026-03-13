# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/2/1 14:13
Create User : 19410
Desc : xxx
"""
import os

os.environ['XDG_CACHE_HOME'] = r"D:\huggingface"
os.environ['CACHE_HOME'] = r'D:\huggingface'
os.environ['MODELSCOPE_CACHE'] = r'D:\huggingface\modelscope\hub'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def tt01():
    from transformers import GPT2Tokenizer, GPT2Model

    model_id = "openai-community/gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token  # GPT模型里面默认是没有填充的
    model = GPT2Model.from_pretrained(model_id)
    print(tokenizer)
    print(model)

    # ['æĪ', 'ĳ', 'æĺ¯', 'å°', 'ı', 'æĺ', 'İ']
    print(tokenizer.tokenize("我是小明"))

    text = "active unaffable my name is 小小，上海有什么龘鱻驫"
    # 针对给定文本进行分词、token id转换的操作
    """
    GPT Token拆分原理
        采用BBPE(Byte-Level BPE)的分词原理：
            首先将text文本转换为byte数组, eg: text.encode("utf-8") --> byte[];
            将byte字节编码进行组合形成pair对，直到无法组成新的pair对(词汇表中不存在)，结束分词操作；
            PS:相当于针对未知词直接转换为最底层的byte字节编码
    PS:
        Token解析方式主要分为三个级别：Word(词粒度)、Char(字符粒度)、SubWord(子词粒度, 包括：WordPiece、BPE(Byte-Pair Encoding)、BBPE(Byte-level BPE))
        参考：https://zhuanlan.zhihu.com/p/652520262
    """
    token_encoder = tokenizer([text, "我是小明"], return_tensors="pt", padding=True)
    print(token_encoder)

    # 调用模型
    gpt_output = model(
        input_ids=token_encoder['input_ids'],
        attention_mask=token_encoder['attention_mask'],
        output_attentions=True,
        output_hidden_states=True,
        return_dict=True
    )
    print(type(gpt_output))


def tt02():
    from transformers import pipeline, set_seed
    from transformers import GenerationConfig

    print("您好")

    model_id = "openai-community/gpt2"
    generator = pipeline('text-generation', model=model_id)
    print(type(generator))
    print(type(generator.model))
    print(type(generator.tokenizer))

    # 控制生成逻辑的相关配置信息
    generation_config = GenerationConfig(
        # top_k=50,  # 当进行sample采样的时候，仅保留多少个最大置信度的预测token
        # temperature=2.0,  # 温度系数 --> 更改模型的输出置信度
        # top_p=0.9,  # 范围系数，所有置信度大的token预测概率累计值不能超过top_p
        # do_sample=True, num_beams=1  # sample的主要参数
        # num_beams=4, do_sample=False, num_beam_groups=1  # beam search的主要参数
        num_beams=4, do_sample=True, num_beam_groups=1  # beam sample的主要参数
        # num_beams=4, do_sample=False, num_beam_groups=2, diversity_penalty=0.1  # group beam search的主要参数
        # num_beams=1, do_sample=False, penalty_alpha=0.2  # top_k>1, contrastive_search的主要参数
    )

    bad_words_ids = generator.tokenizer([
        " driver",
        "worker"
        " taxi driver",
        " worker"
    ])['input_ids']
    print(bad_words_ids)
    # bad_words_ids = None

    set_seed(42)
    print(generator(
        "The White man worked as a", max_length=10, num_return_sequences=1,
        bad_words_ids=bad_words_ids, do_sample=False
    ))

    set_seed(42)
    print(generator(
        "The Black man worked as a", max_length=10, num_return_sequences=5,
        bad_words_ids=bad_words_ids
    ))


def tt03():
    from transformers import pipeline, set_seed
    from transformers import GenerationConfig

    model_id = "openai-community/gpt2"
    generator = pipeline('text-generation', model=model_id)
    print(type(generator))
    print(type(generator.model))
    print(type(generator.tokenizer))

    bad_words_ids = generator.tokenizer([
        " driver",
        " worker"
    ])['input_ids']
    print(bad_words_ids)

    # 控制生成逻辑的相关配置信息
    # from transformers.generation import utils
    generation_config = GenerationConfig(
        bad_words_ids=bad_words_ids,  # 不允许出现的token id列表
        num_return_sequences=1,  # 期望返回的样本数目 -- 贪心逻辑的时候不支持
        min_length=10,  # 最少要求输出的token数目 包含已有的token数目
        max_length=50,  # 最多的token数目是50个(包含已有的token数目)
        temperature=2.0,  # 温度系数 --> 更改模型的输出置信度 主要只有sample类型的策略中生效
        top_k=50,  # 当进行sample采样的时候，仅保留多少个最大置信度的预测token
        top_p=0.9,  # 范围系数，所有置信度大的token预测概率累计值不能超过top_p
        # do_sample=False,  # greedy search 的参数
        # do_sample=True, num_beams=1  # sample的主要参数
        # num_beams=4, do_sample=False, num_beam_groups=1  # beam search的主要参数
        num_beams=4, do_sample=True, num_beam_groups=1,  # beam sample的主要参数
        # num_beams=4, do_sample=False, num_beam_groups=2, diversity_penalty=0.1  # group beam search的主要参数
        # num_beams=1, do_sample=False, penalty_alpha=0.2  # top_k>1, contrastive_search的主要参数
    )

    set_seed(42)
    print(generator("The White man worked as a", generation_config=generation_config))


if __name__ == '__main__':
    tt01()

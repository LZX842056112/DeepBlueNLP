# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/2/1 14:13
Create User : 19410
Desc : xxx
"""
import os

os.environ['XDG_CACHE_HOME'] = r"D:\cache"
os.environ['CACHE_HOME'] = r'D:\cache'
os.environ['MODELSCOPE_CACHE'] = r'D:\cache\modelscope\hub'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


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
    tt03()

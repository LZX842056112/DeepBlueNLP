# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/31 16:30
Create User : 19410
Desc : xxx

https://modelscope.cn/models/iic/ChatPLUG-240M


NOTE: 如果没有科学上网的同学，运行当前代码是会直接失败的(网络异常)，需要进行文件内容的修改：
-1. 首先找到 ChatPLUG-240M 模型所在的文件夹路径； eg: D:\huggingface\modelscope\hub\models\damo\ChatPLUG-240M
-2. 修改模型文件夹下的config.json文件的内容，将 encoder_pth 的参数值直接修改为本地的bert模型路径
    "encoder_pth": "bert-base-chinese",
    --->
    "encoder_pth": "D:\\huggingface\\huggingface\\hub\\models--bert-base-chinese",
-3. 如果还是不行，那么直接科学上网

"""
import json
import os
import random

os.environ['XDG_CACHE_HOME'] = r"D:\cache"
os.environ['CACHE_HOME'] = r'D:\cache'
os.environ['MODELSCOPE_CACHE'] = r'D:\cache\modelscope\hub'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch


def interface01():
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.models import Model

    from modelscope.pipelines.nlp.fid_dialogue_pipeline import FidDialoguePipeline
    from modelscope.models.nlp.fid_plug.text_generation import PlugV2FidChat
    from modelscope.preprocessors.nlp.text_generation_preprocessor import TextGenerationTransformersPreprocessor
    """
    <class 'modelscope.pipelines.nlp.fid_dialogue_pipeline.FidDialoguePipeline'>
    <class 'modelscope.models.nlp.fid_plug.text_generation.PlugV2FidChat'>
    <class 'modelscope.preprocessors.nlp.text_generation_preprocessor.TextGenerationTransformersPreprocessor'>
    """

    model_id = 'damo/ChatPLUG-240M'
    # device可以设置为cpu, cuda, gpu, gpu:X 或 cuda:X
    pipeline_ins = pipeline(
        Tasks.fid_dialogue, model=model_id, model_revision='v1.0.1', device='cpu'
    )
    print(type(pipeline_ins))
    print(type(pipeline_ins.model))
    print(type(pipeline_ins.preprocessor))
    print(pipeline_ins.model)

    # bot_profile: 模型的设定 --- 模型的背景
    # history: 历史对话 + 当前问题
    input = {
        "bot_profile": "我是达摩院的语言模型ChatPLUG， 是基于海量数据训练得到。",
        "history": "你好[SEP]你好!很高兴与你交流![SEP]你叫什么名字[SEP]我是达摩院的语言模型ChatPLUG[SEP]狂飙的导演是谁呀"
    }

    # 数据预处理设置
    preprocess_params = {
        'max_encoder_length': 380,  # encoder最长输入长度
        'context_turn': 3  # context最长轮数
    }

    # 解码策略，默认为sampling
    forward_params = {
        'min_length': 10,
        'max_length': 512,
        'num_beams': 1,
        'temperature': 0.8,
        'do_sample': True,
        'early_stopping': True,
        'top_k': 50,
        'top_p': 0.8,
        'repetition_penalty': 1.2,
        'length_penalty': 1.2,
        'no_repeat_ngram_size': 6
    }

    kwargs = {
        'preprocess_params': preprocess_params,  # 文本处理参数
        'forward_params': forward_params  # 模型推理生成参数
    }

    result = pipeline_ins(input, **kwargs)

    print(result)

    print(pipeline_ins(
        input={
            "history": "你好[SEP]你好!很高兴与你交流![SEP]你叫什么名字",
            "bot_profile": "我是达摩院的语言模型ChatPLUG， 是基于海量数据训练得到。"
        },
        **kwargs
    ))
    print(pipeline_ins(
        input={
            "history": "你好[SEP]你好!很高兴与你交流![SEP]你叫什么名字",
            "bot_profile": "我是小明同学训练的ChatXXX模型，是基于大量数据进行微调得到的。"
        },
        **kwargs
    ))
    print(pipeline_ins(
        input={
            "history": "你好[SEP]你好!很高兴与你交流![SEP]帮忙写一首关于夜色的诗歌",
            "bot_profile": "我是达摩院的语言模型ChatPLUG， 是基于海量数据训练得到。"
        },
        **kwargs
    ))

    print(pipeline_ins(
        input={
            "history": "你好[SEP]你好!很高兴与你交流![SEP]狂飙的导演是谁呀",
            "bot_profile": "我是达摩院的语言模型ChatPLUG， 是基于海量数据训练得到。"
        },
        **kwargs
    ))

    print("=" * 100)


def interface02():
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.models import Model

    """
    <class 'modelscope.pipelines.nlp.fid_dialogue_pipeline.FidDialoguePipeline'>
    <class 'modelscope.models.nlp.fid_plug.text_generation.PlugV2FidChat'>
    <class 'modelscope.preprocessors.nlp.text_generation_preprocessor.TextGenerationTransformersPreprocessor'>
    -1. 将编码器的一条原始文本(当前用户问题)结合模板生成m条文本；
    -2. 针对m条文本进行token分词操作；
    -3. 将分词之后的m条文本输入到bert/transformer encoder模型中，得到输出向量[m,768] 768就是默认的bert模型hidden_size配置大小
    -4. 将提取出来的特征还原成特征向量[m,768] --> [1, m*768] ---> 只有一条样本了
    -5. 基于search的方式进行token的生成(transformer decoder模块)


context_template = '假设我和你正在进行对话，请你给我得体、准确、友好的回复。以下是我们的对话内容。{context}'
history_template = '假设我和你正在进行对话，请你给我得体、准确、友好的回复。以下是我们的对话内容。{context}' \
                   '#以下是在此之前我们的对话内容，可作为回复时的参考。{history}'
knowledge_template = '假设我和你正在进行对话，请你给我得体、准确、友好的回复。以下是我们的对话内容。{context}' \
                     '#以下是和对话相关的知识，请你参考该知识进行回复。{knowledge}'
user_profile_template = '假设我和你正在进行对话，请你给我得体、准确、友好的回复。以下是我们的对话内容。{context}' \
                        '#假设以下是你对我所了解的信息，请你参考该信息并避免你的回复和该信息矛盾，信息如下：{user_profile}'
bot_profile_template = '假设我和你正在进行对话，请你给我得体、准确、友好的回复。以下是我们的对话内容。{context}' \
                       '#假设以下是你的人物设定，请你参考该信息并避免你的回复和该信息矛盾，信息如下：{bot_profile}'

    """

    from modelscope.pipelines.nlp.fid_dialogue_pipeline import FidDialoguePipeline
    from modelscope.models.nlp.fid_plug.text_generation import PlugV2FidChat
    from modelscope.preprocessors.nlp.text_generation_preprocessor import TextGenerationTransformersPreprocessor

    model_id = 'damo/ChatPLUG-240M'
    # device可以设置为cpu, cuda, gpu, gpu:X 或 cuda:X
    pipeline_ins = pipeline(
        Tasks.fid_dialogue, model=model_id, model_revision='v1.0.1', device='cpu'
    )
    print(type(pipeline_ins))
    print(type(pipeline_ins.model))
    print(type(pipeline_ins.preprocessor))
    # print(pipeline_ins.model)

    # 支持输入多段外部知识文本，进行知识增强
    know_list = [
        "《狂飙》由徐纪周执导的。《狂飙》的导演徐纪周也是编剧之一，代表作品有《永不磨灭的番号》《特战荣耀》《心理罪之城市之光》《杀虎口》《胭脂》等",
        "《狂飙》（The Knockout）是一部由 张译、张颂文、李一桐、张志坚 领衔主演，韩童生 特邀主演，吴健、郝平 友情出演，高叶、贾冰、李健 主演，徐纪周 执导，朱俊懿、徐纪周 担任总编剧的 刑侦",
        "狂飙是由中央政法委宣传教育局，中央政法委政法信息中心指导，爱奇艺，留白影视出品，徐纪周执导，张译，李一桐，张志坚领衔主演的刑侦剧。不是。是徐纪周，1976年12月19日出生，毕业于中央戏剧"
    ]

    input = {
        # "history": "你好[SEP]你好!很高兴与你交流![SEP]狂飙的导演是谁呀[SEP]《狂飙》的导演是徐纪周。[SEP]那主演是谁呢[SEP]《狂飙》的主演有张译、张颂文、李一桐、张志坚。[SEP]《胭脂》是不是也是他导演的",
        # "history": "你好[SEP]你好!很高兴与你交流![SEP]狂飙的导演是谁呀[SEP]《狂飙》的导演是徐纪周。[SEP]那主演是谁呢[SEP]《狂飙》的主演有张译、张颂文、李一桐、张志坚。[SEP]《三体》是不是也是徐纪周导演的",
        # "history": "你好[SEP]你好!很高兴与你交流![SEP]狂飙的导演是谁呀[SEP]《狂飙》的导演是徐纪周。[SEP]那主演是谁呢[SEP]《狂飙》的主演包括张译、张颂文、李一桐、张志坚等。[SEP]那他还有什么作品吗",
        "history": "你好[SEP]你好!很高兴与你交流![SEP]你叫什么名字[SEP]我是达摩院的语言模型ChatPLUG[SEP]狂飙的导演是谁呀",
        # "history": "《三体》是不是也是徐纪周导演的",
        "bot_profile": "我是达摩院的语言模型ChatPLUG， 是基于海量数据训练得到。如果问题无法得到明确的返回结果的话，直接返回文本:我不太清楚",
        "knowledge": "[SEP]".join(know_list)  # 外部知识
    }

    # 数据预处理设置
    preprocess_params = {
        'max_encoder_length': 512,  # encoder最长输入长度
        'context_turn': 3  # context最长轮数
    }

    # 解码策略，默认为sampling
    forward_params = {
        'min_length': 1,
        'max_length': 512,
        'num_beams': 1,
        'temperature': 0.8,
        'do_sample': True,
        'early_stopping': True,
        'top_k': 50,
        'top_p': 0.8,
        'repetition_penalty': 1.2,
        'length_penalty': 1.2,
        'no_repeat_ngram_size': 6
    }

    kwargs = {
        'preprocess_params': preprocess_params,
        'forward_params': forward_params
    }

    result = pipeline_ins(input, **kwargs)
    print(result)

    result = pipeline_ins(input={
        "history": "你好[SEP]你好!很高兴与你交流![SEP]你叫什么名字[SEP]我是达摩院的语言模型ChatPLUG[SEP]狂飙的导演还拍摄过什么电影吗",
        "bot_profile": "我是达摩院的语言模型ChatPLUG， 是基于海量数据训练得到。如果问题无法得到明确的返回结果的话，直接返回文本:我不太清楚",
        "knowledge": "[SEP]".join(know_list)  # 外部知识
    }, **kwargs)
    print(result)


if __name__ == '__main__':
    # interface01()
    interface02()

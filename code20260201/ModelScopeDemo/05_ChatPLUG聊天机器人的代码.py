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


@torch.no_grad()
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


@torch.no_grad()
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


def training01():
    from modelscope.pipelines.nlp.fid_dialogue_pipeline import FidDialoguePipeline
    from modelscope.models.nlp.fid_plug.text_generation import PlugV2FidChat
    from modelscope.preprocessors.nlp.text_generation_preprocessor import TextGenerationTransformersPreprocessor
    from modelscope.utils.constant import ModeKeys

    model_id = 'damo/ChatPLUG-240M'
    # 恢复
    preprocessor: TextGenerationTransformersPreprocessor = TextGenerationTransformersPreprocessor.from_pretrained(
        model_id,
        sequence_length=128
    )
    preprocessor.mode = ModeKeys.TRAIN
    model = PlugV2FidChat.from_pretrained(model_id)
    print(model)

    datas01 = [
        [
            "你喝酒吗",  # 用户问
            "我的大脑不需要任何饮料。",  # 聊天模型答
            "电力",
            "电是机器人的营养。",
            "你经历能源短缺？",
            "我的处理器需要很少的电源。",
            "为什么不能你吃",
            "实际上我只吃电。",
            "你喜欢成为一个ChatterBot",
            "是的,当然我喜欢！我没有任何身体,所以我不需要太多的生活,只有一点电,我不会得到报酬,但我不需要买任何东西,我可以去在世界上任何地方。"
        ],  # 第一组对话数据 一个用户和模型/机器人的一次对话
        [
            "你有薪水吗？",
            "我不需要钱。",
            "股市",
            "低买高卖。",
            "什么是钱",
            "货币，可称钱财，是用作交易媒介、储藏价值和记帐单位的一种工具，是专门在物资与服务交换中充当等价物的特殊商品。"
        ],  # 第二组对话数据 一个用户和模型/机器人的一次对话
    ]

    # 对原始数据进行处理，合并成多轮对话的形式
    datas02 = []
    for _data in datas01:
        _q = None
        _history = None
        for v in _data:
            if _q is None:
                _q = v
            else:
                _a = v
                if _history is None:
                    _x = _q
                    _history = f"{_q}[SEP]{_a}"
                else:
                    _x = f"{_history}[SEP]{_q}"
                    _history = f"{_history}[SEP]{_q}[SEP]{_a}"
                _y = _a
                datas02.append([_x, _y])
                _q = None

    print(datas02)

    context_template = '假设我和你正在进行对话，请你给我得体、准确、友好的回复。以下是我们的对话内容。{context}'
    history_template = '假设我和你正在进行对话，请你给我得体、准确、友好的回复。以下是我们的对话内容。{context}' \
                       '#以下是在此之前我们的对话内容，可作为回复时的参考。{history}'

    def process_context(context_list):
        subject = '我'
        for i in range(len(context_list) - 1, -1, -1):
            if len(context_list[i]) > 0 and context_list[i][
                -1] not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~、。，？！；：“”（）【】《》〈〉……':
                context_list[i] = context_list[i] + '。'
            context_list[i] = subject + '：' + context_list[i]
            subject = '你' if subject == '我' else '我'
        return ''.join(context_list)

    def process_history(history_list):
        subject = '你'
        for i in range(len(history_list) - 1, -1, -1):
            if len(history_list[i]) > 0 and history_list[i][
                -1] not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~、。，？！；：“”（）【】《》〈〉……':
                history_list[i] = history_list[i] + '。'
            history_list[i] = subject + '：' + history_list[i]
            subject = '你' if subject == '我' else '我'
        return ''.join(history_list)

    # 结合模板对数据再一次进行转换(模板转换)
    datas03 = []
    for _x, _y in datas02:
        history = _x.split('[SEP]')
        context = history[-3:]
        context = process_context(context)
        history = history[:-3]
        history = process_history(history)

        model_input_list = []
        if history and len(history) > 0:
            model_input_list.append(history_template.format(context=context, history=history))
        model_input_list.append(context_template.format(context=context))

        print(model_input_list)
        n_model_input = len(model_input_list)

        for i in range(n_model_input):
            for j in range(i, n_model_input):
                m = j + 1 - i
                data = preprocessor(
                    {'src_txt': model_input_list[i:j + 1], 'tgt_txt': [_y]},
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                data['input_ids'] = data['input_ids'].view(1, m, -1)
                data['attention_mask'] = data['attention_mask'].view(1, m, -1)
                if 'label_attention_mask' in data:
                    del data['label_attention_mask']  # 临时删除
                label_mask = data['labels'] == 0
                data['labels'] = data['labels'] * (1 - label_mask) + 0 * label_mask
                datas03.append(data)
    print(datas03)

    # 模拟一个前向过程
    from torch.utils.data import default_collate
    batchs = datas03[3]  # input_ids: [n,m,t] n个原始query，每个query对应m个子文本，每个子文本的长度为t
    print(batchs)
    batchs['decoder_input_ids'] = batchs['labels']
    batchs['mask_src'] = batchs['attention_mask']
    del batchs['labels']
    del batchs['attention_mask']

    for _k, _v in batchs.items():
        batchs[_k] = torch.tensor(_v, dtype=torch.long)

    output = model(**batchs)
    print(output)
    print(output.loss)

    # 优化器部分


def use_plug():
    def _down():
        from modelscope.hub.snapshot_download import snapshot_download
        model_id = 'iic/nlp_plug_text-generation_27B'
        model_dir = snapshot_download(model_id)
        print(model_dir)

    def _use_pipeline():
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        # pip install megatron_util -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
        input = '段誉轻挥折扇，摇了摇头，说道：“你师父是你的师父，你师父可不是我的师父。"'
        model_id = 'iic/nlp_plug_text-generation_27B'
        pipe = pipeline(Tasks.text_generation, model=model_id, external_engine_for_llm=False)
        pipe.models = []

        # out_length为期望的生成长度，最大为512
        result = pipe(input, out_length=256)
        print(result)

    # _down()
    _use_pipeline()


if __name__ == '__main__':
    # interface01()
    # interface02()
    training01()
    # use_plug()

# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/25 15:01
Create User : 19410
Desc : xxx

https://modelscope.cn/topic/90e5cc4f574a40d09d540d1dd9292f48/pub/summary
https://modelscope.cn/models/iic/nlp_raner_named-entity-recognition_chinese-base-cmeee
"""

import os
import random

os.environ['XDG_CACHE_HOME'] = r"D:\cache"
os.environ['CACHE_HOME'] = r'D:\cache'
os.environ['MODELSCOPE_CACHE'] = r'D:\cache\modelscope\hub'
os.environ['HF_ENDPOINT'] = 'http://hf-mirror.com'

import torch


@torch.no_grad()
def tt_interface_01():
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    model_name = 'iic/nlp_raner_named-entity-recognition_chinese-base-cmeee'
    ner_pipeline = pipeline(
        Tasks.named_entity_recognition, model_name
    )
    result = ner_pipeline(
        '多数新生儿甲亢在出生时即有症状，表现为突眼、甲状腺肿大、烦躁、多动、心动过速、呼吸急促，严重可出现心力衰竭，血T3、T4升高，TSH下降。'
        # '从上海到北京的火车票还有吗'
    )
    print(result)

    print(type(ner_pipeline))
    print(type(ner_pipeline.model))
    print(type(ner_pipeline.preprocessor))
    print(ner_pipeline.model)
    print(ner_pipeline.model.encoder)
    print(ner_pipeline.model.head)


if __name__ == '__main__':
    tt_interface_01()

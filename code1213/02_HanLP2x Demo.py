# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/13 16:23
Create User : 19410
Desc : 支持两种访问方式两种
方式1：通过客户端，也就是python requests库进行访问（hanlp提供了一个http的请求接口）
方式2: 直接本地加载模型进行预测
"""

import json


def tt_invoker_remote():
    from hanlp_restful import HanLPClient
    from hanlp_common.document import Document
    HanLP = HanLPClient('https://www.hanlp.com/api', auth=None, language='zh')  # auth不填则匿名，zh中文，mul多语种

    # r = HanLP.parse(
    #     "2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。"
    # )
    # print(r)
    # print(type(r))

    r = HanLP.semantic_textual_similarity([
        ('看图猜一电影名', '看图猜电影'),
        ('无线路由器怎么无线上网', '无线上网卡和无线路由器怎么用'),
        ('北京到上海的动车票', '上海到北京的动车票'),
    ])
    print(r)
    print(type(r))


def tt_invoker_remote_with_requests():
    import requests

    url = "https://www.hanlp.com/api/parse"  # 对应的是一个web后端接口(hanlp提供的，我们看不到具体的实现)
    form = {
        'text': '2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。',
        'tokens': None,
        'tasks': 'tok',
        # 'tasks': None,
        'skip_tasks': None,
        'language': 'zh'
    }
    print(json.dumps(form, ensure_ascii=False))
    headers = {
        # 'Authorization': f'Basic {_auth}'
    }
    response = requests.post(url, json=form, headers=headers)
    result = json.loads(response.text)
    print(result)


def tt_invoker_local():
    import hanlp
    from hanlp.components.tokenizers.transformer import TransformerTaggingTokenizer
    from hanlp.components.taggers.transformers.transformer_tagger import TransformerTaggingModel
    from hanlp.components.ner.transformer_ner import TransformerNamedEntityRecognizer
    from hanlp.components.taggers.transformers.transformer_tagger import TransformerTaggingModel

    # 模型只能够契合训练数据范围的数据应用推理
    # 实际工作中一般不太会从0训练模型 --> 从其它某个训练好的模型参数上继续微调训练

    # 分词模型
    tok_hanlp = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
    r = tok_hanlp([
        '2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。',
        '阿婆主来到北京立方庭参观自然语义科技公司。'
    ])
    print(r)
    print(tok_hanlp)
    print(type(tok_hanlp))

    # 命名实体识别模型
    ner_hanlp = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
    ner_r = ner_hanlp(r)
    print(ner_r)
    print(type(ner_hanlp))


if __name__ == '__main__':
    # tt_invoker_remote()
    # tt_invoker_remote_with_requests()
    tt_invoker_local()

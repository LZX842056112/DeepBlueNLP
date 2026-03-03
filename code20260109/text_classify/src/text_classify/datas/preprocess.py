# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/24 21:17
Create User : 19410
Desc : 数据的预处理相关的操作：包括词典的构建....
"""
from .utils import split_text_to_tokens
from ..utils import save_json


def intention_process(intention_data_file, token2id_file, label2id_file):
    """
    意图原始数据的解析构造
    :return:
    """
    import pandas as pd

    df = pd.read_csv(intention_data_file, sep="\t", header=None, names=['text', 'label'])
    token2cnt = {}  # 以token字符串为key，以该token出现的次数为value
    label2cnt = {}  # 以label字符串为key，以该label出现的次数为value
    text_lens = []
    for items in df.iterrows():
        text = items[1]['text'].strip()
        label = items[1]['label'].strip()
        tokens = split_text_to_tokens(text)
        for token in tokens:
            token2cnt[token] = token2cnt.get(token, 0) + 1
        label2cnt[label] = label2cnt.get(label, 0) + 1
        text_lens.append(len(tokens))

    # 基于单词的数量构建词典
    token2ids = {
        "<PAD>": 0,
        "<UNK>": 1
    }
    for token, cnt in token2cnt.items():
        if cnt < 3:
            continue  # 一般情况下出现次数太少的单词直接过滤
        token2ids[token] = len(token2ids)
    save_json(token2id_file, token2ids)

    label2ids = {}
    for label, cnt in label2cnt.items():
        label2ids[label] = len(label2ids)
    save_json(label2id_file, label2ids)


def senti_corp_process(sentiment_data_file, token2id_file, label2id_file):
    """
    情感数据的解析
    :return:
    """
    import pandas as pd

    df = pd.read_csv(sentiment_data_file, sep=",")
    df = df[['label', 'review']]
    df.columns = ['label', 'text']

    token2cnt = {}  # 以token字符串为key，以该token出现的次数为value
    label2cnt = {}  # 以label字符串为key，以该label出现的次数为value
    text_lens = []
    for items in df.iterrows():
        text = str(items[1]['text']).strip()
        label = str(items[1]['label']).strip()
        tokens = split_text_to_tokens(text)
        for token in tokens:
            token2cnt[token] = token2cnt.get(token, 0) + 1
        label2cnt[label] = label2cnt.get(label, 0) + 1
        text_lens.append(len(tokens))

    # 基于单词的数量构建词典
    token2ids = {
        "<PAD>": 0,
        "<UNK>": 1
    }
    for token, cnt in token2cnt.items():
        if cnt < 3:
            continue  # 一般情况下出现次数太少的单词直接过滤
        token2ids[token] = len(token2ids)
    save_json(token2id_file, token2ids)

    label2ids = {}
    for label, cnt in label2cnt.items():
        label2ids[label] = len(label2ids)
    save_json(label2id_file, label2ids)

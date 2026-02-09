# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/7 10:38
Create User : 19410
Desc : jieba分词内部主要基于词典匹配实现，首先获取所有可能的分词方式，然后在所有分词方式选择最有可能的分词结果
"""
import os

import jieba

jieba.load_userdict("./text_classify.dict")


def split_text(text):
    return jieba.lcut(text) # 分词
    # return list(text)  # 分字 -- 直接将字作为词存在


def t0(in_file, out_file):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(in_file, "r", encoding="utf-8") as reader:
        with open(out_file, "w", encoding="utf-8") as writer:
            for line in reader:
                text, label = line.strip().split("\t")

                tokens = split_text(text)

                writer.writelines(f"{' '.join(tokens)}\t{label}\n")


if __name__ == '__main__':
    t0(
        in_file="../datas/text_classify/train.csv",
        out_file="../datas/text_classify/train_tokens.csv"
    )

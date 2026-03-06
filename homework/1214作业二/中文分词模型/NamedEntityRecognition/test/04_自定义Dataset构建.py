# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/10 16:47
Create User : 19410
Desc : xxx
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

def t0():
    from ner.datas.dataset import NerTokenClassifyDataset
    from ner.datas.tokenizer import Tokenizer
    from ner.utils import load_json

    tokenizer = Tokenizer(
        #vocabs=r'./datas/medical/vocab.txt'
        vocabs=r'./datas/vocab.txt'
    )
    label2id = load_json(
        file="./datas/word_segment/label2id.json"
    )
    ds = NerTokenClassifyDataset(
        in_file=r"./datas/word_segment/train.jsonl",
        tokenizer=tokenizer,
        label2id=label2id,
        append_special_token=True,
        max_length=20
    )
    print(ds[0])


if __name__ == '__main__':
    t0()

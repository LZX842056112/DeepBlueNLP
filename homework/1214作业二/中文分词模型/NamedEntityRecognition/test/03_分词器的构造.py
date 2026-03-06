# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/10 15:58
Create User : 19410
Desc : xxx
"""


def t0():
    from ner.datas.tokenizer import Tokenizer

    tokenizer = Tokenizer(
        vocabs=r'./datas/medical/vocab.txt'
    )
    result = tokenizer(
        "患者因罹患“胃癌”在我院予行全麻上胃癌根治术",
        append_cls=True,
        append_sep=True
    )
    print(result)


if __name__ == '__main__':
    t0()

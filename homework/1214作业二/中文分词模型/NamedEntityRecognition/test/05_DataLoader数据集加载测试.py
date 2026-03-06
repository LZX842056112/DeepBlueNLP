# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/11 11:23
Create User : 19410
Desc : xxx
"""


def t0():
    from ner.datas.dataset import NerTokenClassifyDataset, build_dataloader
    from ner.datas.tokenizer import Tokenizer
    from ner.utils import load_json

    tokenizer = Tokenizer(
        vocabs=r'./datas/medical/vocab.txt'
    )
    label2id = load_json(
        file="./datas/medical/label2id.json"
    )
    ds = NerTokenClassifyDataset(
        in_file=r"./datas/medical/training.txt",
        tokenizer=tokenizer,
        label2id=label2id,
        append_special_token=True,
        max_length=512
    )
    dataloader = build_dataloader(ds, 2, shuffle=True)
    for batch in dataloader:
        print(batch)
        break


if __name__ == '__main__':
    t0()

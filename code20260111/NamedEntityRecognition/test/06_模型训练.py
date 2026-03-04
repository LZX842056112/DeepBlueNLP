# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/11 14:01
Create User : 19410
Desc : xxx
"""

import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def t0():
    import sys

    sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

    from ner.trainer.trainer import Trainer
    from ner.config import Config
    from ner.datas.tokenizer import Tokenizer

    bert_path = r"D:\cache\huggingface\hub\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
    tokenizer = Tokenizer(
        vocabs=r'./datas/medical/vocab.txt'
    )

    cfg = Config(
        output_dir="./output/medical/",
        tokenizer=tokenizer,  # 分词器
        label2id="./datas/medical/label2id.json",  # 类别标签名称到id的映射

        train_file="./datas/medical/training.txt",  # 训练数据对应文件
        eval_file="./datas/medical/test.json",  # 模型评估数据对应文件

        total_epoch=10,
        batch_size=8,  # 批次大小
        lr=0.01,  # 模型训练学习率

        # network_type='lstm',  # 网络类型 可选lstm、bert
        network_type='bert',  # 网络类型 可选lstm、bert
        lstm_layers=3,  # LSTM的层数
        lstm_hidden_size=768,
        bert_path=bert_path,  # Bert模型迁移路径
        max_length=128,
        freeze=True,  # 给定迁移模型的时候冻结参数
        device="cuda"
    )
    trainer = Trainer(config=cfg)

    trainer.training()


if __name__ == '__main__':
    t0()

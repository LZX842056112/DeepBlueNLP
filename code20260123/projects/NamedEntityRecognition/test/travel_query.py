# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/17 11:44
Create User : 19410
Desc : 交通查询命名实体识别
"""

import json
import os
import random
import sys

project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(project_root, "src"))


def t0():
    """
        如果需要自己标注，可以考虑使用Label Studio 进行标注
    将原始数据进行转换 --> 转换为当前训练点支持的结构
    :return:
    """
    in_file = "./datas/travel_query/project-2-at-2026-01-17-03-18-55b6bb0c.json"
    training_file = "./datas/travel_query/training.txt"
    test_file = "./datas/travel_query/test.txt"

    # 创建输出文件夹
    os.makedirs(os.path.dirname(training_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_file), exist_ok=True)

    new_records = []
    with open(in_file, "r", encoding="utf-8") as reader:
        records = json.load(reader)
        for record in records:
            entities = []
            for label in record['label']:
                start_pos = label['start']  # 包含
                end_pos = label['end']  # 不包含
                for label_type in label['labels']:
                    entities.append({
                        "label_type": label_type,
                        "overlap": 0,
                        "start_pos": start_pos,  # 包含
                        "end_pos": end_pos  # 不包含
                    })
            new_record = {
                'originalText': record['text'],
                "entities": entities
            }
            new_records.append(new_record)

    with (open(training_file, "w", encoding="utf-8") as train_write,
          open(test_file, "w", encoding="utf-8") as test_write):
        for record in new_records:
            record = json.dumps(record, ensure_ascii=False)
            if random.random() < 0.9:
                train_write.writelines(f"{record}\n")
            else:
                test_write.writelines(f"{record}\n")


def t1():
    """
    训练
        1. 基于数据构造训练依赖的元数据, eg: 标签映射mapping
        2. 执行下列训练代码--给定好对应的各种参数
    :return:
    """
    from ner.trainer.trainer import Trainer
    from ner.config import Config
    from ner.datas.tokenizer import Tokenizer

    bert_path = r"D:\huggingface\huggingface\hub\models--bert-base-chinese"
    if not os.path.exists(bert_path):
        bert_path = "bert-base-chinese"

    tokenizer = Tokenizer(
        vocabs=r'./datas/vocab.txt'
    )

    data_root_dir = r"./datas/travel_query"
    cfg = Config(
        output_dir="./output/travel_query/",
        tokenizer=tokenizer,  # 分词器
        label2id=os.path.join(data_root_dir, "label2id.json"),  # 类别标签名称到id的映射

        train_file=os.path.join(data_root_dir, "training.txt"),  # 训练数据对应文件
        eval_file=os.path.join(data_root_dir, "test.txt"),  # 模型评估数据对应文件

        total_epoch=3,
        batch_size=8,  # 批次大小
        lr=0.01,  # 模型训练学习率

        # network_type='lstm',  # 网络类型 可选lstm、bert
        network_type='bert',  # 网络类型 可选lstm、bert
        lstm_layers=3,  # LSTM的层数
        lstm_hidden_size=768,
        bert_path=bert_path,  # Bert模型迁移路径
        max_length=512,
        freeze=True,  # 给定迁移模型的时候冻结参数
        device="cuda",
        max_no_improved_epoch=20
    )
    trainer = Trainer(config=cfg)

    trainer.training()


def t2():
    import requests

    url = "http://127.0.0.1:9001/predict"

    def tt_get():
        text = "帮我查一下最近去湘潭的高铁什么时候出发"
        text = "今年暑假，我准备去西安、新疆这两个地方去玩，但是不知道能不能从上海坐火车过去?"
        response = requests.get(
            url, json={'text': text}
        )
        if response.status_code == 200:
            print("访问服务器成功GET")
            # 明确的知道服务器返回的是json格式，那么可以直接调用json()这个方法转换为字典对象
            result = response.json()
            print(result)
            print(type(result))
            if result['code'] == 0:
                print(f"调用模型成功GET，结果为:{result['data']}")
            else:
                print(f"调用模型服务器处理异常，异常信息为:{result['msg']}")
        else:
            print("访问服务器失败：网络异常等等")

    def tt_post():
        # NOTE: 在fastapi框架中，post请求的参数必须通过json参数给定
        text = "帮我查一下最近去湘潭的高铁什么时候出发"
        text = "去虹桥火车站最快的路线是什么"
        text = "从我家到嘉兴北站怎么走"
        text = "附近有什么好玩的游乐场吗"
        text = "今年暑假，我准备去西安、新疆这两个地方去玩，帮我查一下从长沙的火车有那些"

        response = requests.post(
            url,
            json={'text': text},
        )
        if response.status_code == 200:
            print("访问服务器成功POST")
            # 明确的知道服务器返回的是json格式，那么可以直接调用json()这个方法转换为字典对象
            result = response.json()
            print(result)
            print(type(result))
            if result['code'] == 0:
                print(f"调用模型成功POST，结果为:{result['data']}")
            else:
                print(f"调用模型服务器处理异常，异常信息为:{result['msg']}")
        else:
            print("访问服务器失败：网络异常等等")

    tt_get()
    tt_post()


if __name__ == '__main__':
    # t0()  # 将原始数据转换
    # t1()  # 模型训练
    t2()  # 访问接口模型

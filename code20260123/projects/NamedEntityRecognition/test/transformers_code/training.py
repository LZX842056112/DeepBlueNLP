# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/17 14:42
Create User : 19410
Desc : 基于transformers框架的命名实体识别模型训练
"""
import os

import torch

os.environ['HF_ENDPOINT'] = 'http://hf-mirror.com'
os.environ['XDG_CACHE_HOME'] = r'D:\huggingface'

from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers import BertForTokenClassification, BertTokenizer
from datasets import load_dataset, DatasetDict

from ner.metrics import build_metric_func
from ner.utils import load_json
from ner.datas.tokenizer import Tokenizer
from ner.datas.utils import parse_record


def run():
    output_dir = "./output/travel_query/modules"
    logging_dir = "./output/travel_query/logs"
    data_dir = "../datas/travel_query"
    bert_path = r"D:\huggingface\huggingface\hub\models--bert-base-chinese"
    if not os.path.exists(bert_path):
        bert_path = "bert-base-chinese"

    # 0. 加载元数据/依赖数据
    label2id = load_json(os.path.join(data_dir, "label2id.json"))

    # 1. 数据加载
    # <class 'datasets.dataset_dict.DatasetDict'>
    # <class 'datasets.arrow_dataset.Dataset'>
    dataset: DatasetDict = load_dataset(
        "json",
        data_dir=data_dir,
        data_files={
            "train": "training.txt",
            "test": "test.txt"
        }
    )
    print(dataset)
    # print(type(dataset))
    # print(type(dataset['train']))
    # print(dataset['train'][0])
    # print(dataset['test'][0])

    # 2. 模型迁移恢复(分词器 + 模型)
    bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
    tokenizer: Tokenizer = Tokenizer(
        vocabs=bert_tokenizer.vocab,
        unk_token=bert_tokenizer.unk_token,
        pad_token=bert_tokenizer.pad_token,
        cls_token=bert_tokenizer.cls_token,
        sep_token=bert_tokenizer.sep_token
    )
    bert_model = BertForTokenClassification.from_pretrained(
        bert_path,
        num_labels=len(label2id),
        id2label={_id: _label for _label, _id in label2id.items()},
        label2id=label2id,
        weights_only=False  # 当前参数在某些版本中需要加入，某些版本不需要 -- 具体根据异常来决定
    )
    # for param in bert_model.bert.parameters():
    #     param.requires_grad = False
    max_length = bert_model.config.max_position_embeddings
    print(bert_model)

    # 3. 数据解析转换
    def process_example(examples):
        """
        :param examples:
            Dict[str,Any]  -- key就是dataset的列名称, value就是对应的值 ---> 当map函数的batched参数为False的时候
            Dict[str, List[Any]] --- key就是dataset的列名称, value是一个批次的所有样本的对应值组成的list ---> 当map函数的batched参数为True的时候
        :return:
        """
        text = examples['originalText'][0]  # eg: "xxxx"
        entities = examples['entities'][0]  # list[xxxx]

        _iter = parse_record(
            text, entities, tokenizer,
            True, max_length, label2id,
            return_pt=False
        )

        datas = {}
        for data in _iter:
            data_keys = data.keys()
            if len(datas) == 0:
                for _key in data_keys:
                    datas[_key] = []
            for _key in data_keys:
                datas[_key].append(data[_key])
        return datas

    dataset = dataset.map(
        function=process_example,
        batched=True,  # 是否批次数据转换 --> 当前强制要求必须为True
        batch_size=1,  # 当batched=True的时候，表示一次将多少个样本组合成一个批次 --> 当前强制要求是1
        remove_columns=["originalText", "entities"],  # 当前强制要求必须给定
        num_proc=None  # 数据多进程处理的进程数目
    )
    dataset = dataset.select_columns(['token_ids', 'token_masks', 'label_ids'])
    dataset = dataset.rename_columns({
        'token_ids': 'input_ids',
        'token_masks': 'attention_mask',
        'label_ids': 'labels'
    })

    # print(dataset)
    # print(dataset['train'][0])

    # 4. 定义训练参数/对象
    _metrics_func = build_metric_func(
        label_id2names=bert_model.config.id2label
    )

    @torch.no_grad()
    def compute_metrics(eval_pred):
        # PS: 默认情况下，会自动将多个批次的数据合并成一个tensor/ndarray对象
        predictions = torch.tensor(eval_pred.predictions)  # [bs,t,c] 预测置信度
        label_ids = torch.tensor(eval_pred.label_ids)  # [bs,t] 实际标签
        metric_result = _metrics_func([predictions], None, [label_ids])
        return metric_result

    training_args = TrainingArguments(
        output_dir=output_dir,  # 模型保存路径
        overwrite_output_dir=True,
        num_train_epochs=3,  # 训练轮数
        per_device_train_batch_size=4,  # 单设备训练批次大小（视GPU内存调整）
        per_device_eval_batch_size=4,  # 单设备验证批次大小
        gradient_accumulation_steps=4,  # 梯度累积（显存不足时增大，等效于增大batch_size）
        eval_strategy="epoch",  # 每轮结束后验证
        save_strategy="epoch",  # 每轮结束后保存模型
        logging_dir=logging_dir,  # 日志路径
        logging_steps=10,
        learning_rate=5e-5,  # 学习率（GPT类模型通常用2e-5 ~ 5e-5）
        weight_decay=0.01,  # 权重衰减（正则化）
        fp16=True,  # 启用混合精度训练（需GPU支持）
        load_best_model_at_end=True,  # 训练结束后加载最佳模型
        metric_for_best_model="entity_f1",  # 以准确率为判断标准 默认为损失
        greater_is_better=True,  # metric_for_best_model越高越好（损失则设为 False， 默认为损失）
    )

    # 数据填充对象--> 将多条样本数据组合成一个批次
    data_collator = DataCollatorForTokenClassification(tokenizer=bert_tokenizer)

    # 初始化Trainer
    trainer = Trainer(
        model=bert_model,
        args=training_args,
        data_collator=data_collator,  # 传入数据整理器
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,  # 评估指标
    )

    # 5. 开始训练
    trainer.train()

    # 6. 模型保存(最终的最优模型保存)
    trainer.save_model(os.path.join(training_args.output_dir, "./final-best-model"))


if __name__ == '__main__':
    run()

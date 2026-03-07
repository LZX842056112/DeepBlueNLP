# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/16 20:55
Create User : 19410
Desc : xxx
"""

import os
import random

os.environ['HF_ENDPOINT'] = 'http://hf-mirror.com'
# 下列两个任选一个
# os.environ['TRANSFORMERS_CACHE'] = '/opt/work/huggingface/hub'
# os.environ['XDG_CACHE_HOME'] = '/opt/work'
os.environ['XDG_CACHE_HOME'] = r'D:\huggingface'


def training():
    from transformers import DataCollatorWithPadding
    from transformers import TrainingArguments, Trainer
    from transformers import BertForSequenceClassification, BertTokenizer
    from datasets import load_dataset
    import torch
    import numpy as np

    # 0. 加载标签列表
    labels = [
        'Alarm-Update', 'Audio-Play', 'Calendar-Query',
        'FilmTele-Play', 'HomeAppliance-Control', 'Music-Play',
        'Other', 'Radio-Listen', 'TVProgram-Play',
        'Travel-Query', 'Video-Play', 'Weather-Query'
    ]
    labelname2id = {label_name: label_idx for label_idx, label_name in enumerate(labels)}

    # 1. 加载数据集
    dataset = load_dataset(
        "csv",
        data_dir="datas/text_classify/intention",
        data_files={
            "train": "train.csv"
        },
        sep="\t",
        header=None,
        names=['text', 'label']
    )

    # 2. 模型恢复
    bert_path = "bert-base-chinese"
    bert_path = r"D:\huggingface\huggingface\hub\models--bert-base-chinese"
    # Bert模型迁移
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    # 迁移模型，并且更改最终输出的类别数目
    model = BertForSequenceClassification.from_pretrained(
        bert_path,
        num_labels=len(labelname2id),  # 重新更改标签数目
        id2label=dict(enumerate(labels)),  # 重新更改id和标签的映射mapping
        label2id=labelname2id,  # 重新更改id和标签的映射mapping
        weights_only=False  # 这个参数weights_only在不同的transformers版本中不一定需要
    )
    print(model)

    # 3. 数据转换
    def preprocess_function(examples):
        """
        对单个文本进行分词转换
        """
        text, label = examples['text'], examples['label']
        # 分词转换
        item = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=False
        )

        item["labels"] = torch.tensor(labelname2id[label])
        return item

    # 应用预处理函数（num_proc=4表示多进程加速）
    tokenized_dataset = dataset.map(
        preprocess_function,
        num_proc=None,
        remove_columns=['text', 'label']  # 删除列
    )
    # 数据抽样以及数据分割
    sample_dataset = tokenized_dataset['train']
    train_test_dataset = sample_dataset.filter(lambda t: random.random() < 0.01).train_test_split(
        test_size=0.2,
        seed=24
    )

    # 4. 评估方法的构造
    def compute_metrics(eval_pred):
        # PS: 默认情况下，会自动将多个批次的数据合并成一个tensor/ndarray对象
        label_ids = eval_pred.label_ids  # 评估数据的实际标签值 [bs]
        predictions = eval_pred.predictions  # 评估数据经过模型前向的输出值 [bs,class_num]
        pred_ids = np.argmax(predictions, axis=-1)  # [bs,class_num] -> [bs]
        from sklearn import metrics
        f1 = metrics.f1_score(label_ids, pred_ids, average='micro')
        acc = metrics.accuracy_score(label_ids, pred_ids)
        return {
            "f1": f1,
            "acc": acc
        }

    # 5. 训练参数定义
    # 可能需要安装：pip install transformers[torch]
    training_args = TrainingArguments(
        output_dir="output/bert-finetuned-intent-textclassify/models",  # 模型保存路径
        overwrite_output_dir=True,
        num_train_epochs=3,  # 训练轮数
        per_device_train_batch_size=4,  # 单设备训练批次大小（视GPU内存调整）
        per_device_eval_batch_size=4,  # 单设备验证批次大小
        gradient_accumulation_steps=4,  # 梯度累积（显存不足时增大，等效于增大batch_size）
        eval_strategy="epoch",  # 每轮结束后验证
        save_strategy="epoch",  # 每轮结束后保存模型
        logging_dir="output/bert-finetuned-intent-textclassify/logs",  # 日志路径
        logging_steps=10,
        learning_rate=5e-5,  # 学习率（GPT类模型通常用2e-5 ~ 5e-5）
        weight_decay=0.01,  # 权重衰减（正则化）
        fp16=True,  # 启用混合精度训练（需GPU支持）
        load_best_model_at_end=True,  # 训练结束后加载最佳模型
        metric_for_best_model="f1",  # 以准确率为判断标准 默认为损失
        greater_is_better=True,  # metric_for_best_model越高越好（损失则设为 False， 默认为损失）
    )

    # 数据填充对象--> 将多条样本数据组合成一个批次
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,  # 传入数据整理器
        train_dataset=train_test_dataset["train"],
        eval_dataset=train_test_dataset["test"],
        compute_metrics=compute_metrics,  # 评估指标
    )

    # 开始训练
    trainer.train()

    # 模型保存(最终的最优模型保存)
    trainer.save_model(os.path.join(training_args.output_dir, "./final-best-model"))
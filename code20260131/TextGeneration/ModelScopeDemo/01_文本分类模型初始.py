# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/25 10:27
Create User : 19410
Desc : xxx
pip install modelscope==1.27.1
https://modelscope.cn/models/iic/nlp_structbert_sentiment-classification_chinese-base

"""
import os
import random

os.environ['XDG_CACHE_HOME'] = r"D:\huggingface"
os.environ['CACHE_HOME'] = r'D:\huggingface'
os.environ['MODELSCOPE_CACHE'] = r'D:\huggingface\modelscope\hub'
os.environ['HF_ENDPOINT'] = 'http://hf-mirror.com'

import torch
import torch.nn as nn


@torch.no_grad()
def interface01():
    """
    模型推理使用
    PS: 基于pipeline结构的推理预测
    :return:
    """
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.pipelines.nlp.text_classification_pipeline import TextClassificationPipeline
    from modelscope.preprocessors.nlp.text_classification_preprocessor import TextClassificationTransformersPreprocessor
    from modelscope.models.nlp.task_models.text_classification import ModelForTextClassification

    semantic_cls = pipeline(
        Tasks.text_classification,  # 任务名称
        'damo/nlp_structbert_sentiment-classification_chinese-base'  # 模型名称字符串或者本地路径
    )
    # r = semantic_cls(input='启动的时候很大声音，然后就会听到1.2秒的卡察的声音，类似齿轮摩擦的声音')
    r = semantic_cls(
        # input='启动的时候很大声音，然后就会听到1.2秒的卡察的声音，类似齿轮摩擦的声音。不过其它机器的声音更嘈杂，现在这个机器已经算是优化后的了'
        input="当机器启动的时候，有1到2秒有特别明显的声音，在我们的业务场景中，具有非常明显的提示效果"
    )
    print(r)
    print(type(semantic_cls))
    print(type(semantic_cls.model))
    print(type(semantic_cls.preprocessor))
    print(semantic_cls.model)


@torch.no_grad()
def interface02():
    """
    拆分推理过程
    PS：将pipeline的推理结构拆分开来推理预测
    :return:
    """
    from modelscope.preprocessors.nlp.text_classification_preprocessor import TextClassificationTransformersPreprocessor
    from modelscope.models.nlp.task_models.text_classification import ModelForTextClassification

    model_name = "damo/nlp_structbert_sentiment-classification_chinese-base"
    model_name = "./workspace/text_classify/intention/output_best"

    preprocessor = TextClassificationTransformersPreprocessor.from_pretrained(model_name)
    model = ModelForTextClassification.from_pretrained(model_name)
    model.eval().cpu()

    text = "当机器启动的时候，有1到2秒有特别明显的声音，在我们的业务场景中，具有非常明显的提示效果"
    text = "明天上海温度怎么样？最近感觉都降温了，好冷呀"
    output = preprocessor(text)  # 前处理 --- 分词+token2id
    output = model(**output)  # 前向预测 --- 获取预测各个类别的置信度

    # 后处理 --- 将预测结果转换为需要的数据格式
    logits = output.logits  # 预测置信度
    print(logits)
    print(torch.softmax(logits, dim=-1))  # 预测概率
    print(torch.argmax(logits, dim=-1))  # 预测id
    print(preprocessor.label2id)
    print(preprocessor.id2label)


def training01():
    import os.path as osp
    from modelscope.trainers import build_trainer
    from modelscope.msdatasets import MsDataset
    from modelscope.utils.hub import read_config
    from modelscope.metainfo import Metrics, Hooks
    from modelscope.utils.constant import DownloadMode

    model_id = 'damo/nlp_structbert_sentiment-classification_chinese-base'
    dataset_id = 'jd'

    WORK_DIR = f'workspace/text_classify/{dataset_id}'

    max_epochs = 2

    def cfg_modify_fn(cfg):
        """
        修改训练参数信息
        :param cfg: 本质上是一个字典
        :return:
        """
        cfg.train.dataloader.workers_per_gpu = 0
        cfg.train.dataloader.batch_size_per_gpu = 4
        cfg.evaluation.dataloader.workers_per_gpu = 0
        cfg.evaluation.dataloader.batch_size_per_gpu = 8

        cfg.train.max_epochs = max_epochs
        cfg.train.hooks = [
            {
                'type': 'TextLoggerHook',
                'interval': 100
            },
            {
                "type": "CheckpointHook",
                "interval": 1
            },
            {
                "type": "EvaluationHook",
                "interval": 1
            },
            {
                "type": "BestCkptSaverHook",  # 必须有 EvaluationHook
                "metric_key": "f1"
            }
        ]
        cfg.evaluation.metrics = [Metrics.seq_cls_metric]
        cfg['dataset'] = {
            'train': {
                'labels': ['负面', '正面'],  # 数据分词解析处理的时候对应的标签名称列表
                'first_sequence': 'sentence',  # 第一个文本的列名称(key)
                'label': 'label',  # 标签的列名称(key)
            }
        }
        cfg.train.optimizer.lr = 3e-5
        return cfg

    train_dataset = MsDataset.load(
        dataset_id, namespace='DAMO_NLP', split='train',
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
    ).to_hf_dataset()
    eval_dataset = MsDataset.load(
        dataset_id, namespace='DAMO_NLP', split='validation',
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS
    ).to_hf_dataset()
    print(type(train_dataset))  # 实际上就是transformers库底层依赖的datasets数据加载库的代码逻辑
    # remove useless case
    train_dataset = train_dataset.filter(lambda x: x["label"] != None and x["sentence"] != None)
    eval_dataset = eval_dataset.filter(lambda x: x["label"] != None and x["sentence"] != None)
    # 仅选择部分数据参与模型训练 --> 为了快速跑完整的带
    train_dataset = train_dataset.filter(lambda x: random.random() < 0.001)
    eval_dataset = eval_dataset.filter(lambda x: random.random() < 0.002)

    # map float to index
    def map_labels(examples):
        map_dict = {0: "负面", 1: "正面"}
        examples['label'] = map_dict[int(examples['label'])]
        return examples

    train_dataset = train_dataset.map(map_labels)
    eval_dataset = eval_dataset.map(map_labels)

    kwargs = dict(
        model=model_id,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        work_dir=WORK_DIR,
        cfg_modify_fn=cfg_modify_fn
    )

    from modelscope.trainers.nlp_trainer import NlpEpochBasedTrainer
    trainer = build_trainer(name='nlp-base-trainer', default_args=kwargs)
    print(type(trainer))

    print('===============================================================')
    print('pre-trained model loaded, training started:')
    print('===============================================================')

    trainer.train()

    print('===============================================================')
    print('train success.')
    print('===============================================================')

    for i in range(max_epochs):
        eval_results = trainer.evaluate(f'{WORK_DIR}/epoch_{i + 1}.pth')
        print(f'epoch {i} evaluation result:')
        print(eval_results)

    print('===============================================================')
    print('evaluate success')
    print('===============================================================')


def load_custom_data():
    from modelscope.msdatasets import MsDataset
    from datasets import load_dataset

    ds = MsDataset.load(
        "csv",
        data_files={
            "train": "./datas/text_classify/intention/train.csv",
            "test": "./datas/text_classify/intention/test.csv"
        },
        sep="\t",
        header=None,
        names=["text", "label"]
    )
    train_ds = ds['train'].to_hf_dataset()
    test_ds = ds['test'].to_hf_dataset()
    print(type(train_ds))
    print(train_ds)


def training02():
    import os.path as osp
    from modelscope.trainers import build_trainer
    from modelscope.msdatasets import MsDataset
    from modelscope.utils.hub import read_config
    from modelscope.metainfo import Metrics, Hooks
    from modelscope.utils.constant import DownloadMode

    model_id = 'damo/nlp_structbert_sentiment-classification_chinese-base'

    WORK_DIR = f'workspace/text_classify/intention'

    max_epochs = 2

    def load_state_fn(
            model,
            state_dict,
            prefix=None,
            head_prefix_keys=None,
            local_metadata=None,
            error_msgs=[]
    ):
        print("自定义模型参数恢复逻辑 --> ")
        if prefix is None:
            prefix = ''
        # 比较当前模型和给定的参数列表，确定最终能够恢复的参数
        model_state_dict = model.state_dict()
        restore_state_dict = {}
        for key, value in model_state_dict.items():
            restore_key = prefix + key
            if restore_key in state_dict:
                restore_tensor = state_dict[restore_key]
                if restore_tensor.shape == value.shape:
                    restore_state_dict[key] = restore_tensor
        # 进行参数恢复
        missing_keys, unexpected_keys = model.load_state_dict(restore_state_dict, strict=False)
        print(f"未进行参数恢复的列表为:{missing_keys}")
        print(f"额外多给的参数列表为:{unexpected_keys}")

    def cfg_modify_fn(cfg):
        """
        修改训练参数信息
        :param cfg: 本质上是一个字典
        :return:
        """
        cfg.train.dataloader.workers_per_gpu = 0
        cfg.train.dataloader.batch_size_per_gpu = 4
        cfg.evaluation.dataloader.workers_per_gpu = 0
        cfg.evaluation.dataloader.batch_size_per_gpu = 8

        cfg.train.max_epochs = max_epochs
        cfg.train.hooks = [
            {
                'type': 'TextLoggerHook',
                'interval': 100
            },
            {
                "type": "CheckpointHook",
                "interval": 1
            },
            {
                "type": "EvaluationHook",
                "interval": 1
            },
            {
                "type": "BestCkptSaverHook",  # 必须有 EvaluationHook
                "metric_key": "f1"
            }
        ]
        cfg.evaluation.metrics = [Metrics.seq_cls_metric]
        cfg['dataset'] = {
            'train': {
                'labels': [
                    "Alarm-Update", "Audio-Play", "Calendar-Query",
                    "FilmTele-Play", "HomeAppliance-Control", "Music-Play",
                    "Other", "Radio-Listen", "TVProgram-Play",
                    "Travel-Query", "Video-Play", "Weather-Query"
                ],  # 数据分词解析处理的时候对应的标签名称列表
                'first_sequence': 'text',  # 第一个文本的列名称(key)
                'label': 'label',  # 标签的列名称(key)
            }
        }
        cfg.train.optimizer.lr = 3e-5

        # 通过load_state_fn方法给定模型参数恢复的逻辑
        ## 同步修改 modelscope/utils/checkpoint.py 文件 426行左右的模型恢复加载代码
        """
        # 修改后的内容
        error_msgs = _load_state_dict_into_model(
            model_to_load,
            state_dict,
            start_prefix,
            start_prefix,
            load_state_fn=load_state_fn
        )
        """
        cfg.model.load_state_fn = load_state_fn
        return cfg

    ds = MsDataset.load(
        "csv",
        data_files={
            "train": "./datas/text_classify/intention/train.csv",
            "test": "./datas/text_classify/intention/test.csv"
        },
        sep="\t",
        header=None,
        names=["text", "label"]
    )
    train_dataset = ds['train'].to_hf_dataset()
    eval_dataset = ds['test'].to_hf_dataset()
    # remove useless case
    train_dataset = train_dataset.filter(lambda x: x["label"] != None and x["text"] != None)
    eval_dataset = eval_dataset.filter(lambda x: x["label"] != None and x["text"] != None)
    # 仅选择部分数据参与模型训练 --> 为了快速跑完整个训练过程
    train_dataset = train_dataset.filter(lambda x: random.random() < 0.01)
    eval_dataset = eval_dataset.filter(lambda x: random.random() < 0.01)
    print(train_dataset)
    print(eval_dataset)

    kwargs = dict(
        model=model_id,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        work_dir=WORK_DIR,
        cfg_modify_fn=cfg_modify_fn
    )

    from modelscope.trainers.nlp_trainer import NlpEpochBasedTrainer
    trainer = build_trainer(name='nlp-base-trainer', default_args=kwargs)
    print(type(trainer))

    print('===============================================================')
    print('pre-trained model loaded, training started:')
    print('===============================================================')

    trainer.train()

    print('===============================================================')
    print('train success.')
    print('===============================================================')

    for i in range(max_epochs):
        eval_results = trainer.evaluate(f'{WORK_DIR}/epoch_{i + 1}.pth')
        print(f'epoch {i} evaluation result:')
        print(eval_results)

    print('===============================================================')
    print('evaluate success')
    print('===============================================================')


if __name__ == '__main__':
    # interface01()
    interface02()
    # training01()
    # load_custom_data()
    # training02()

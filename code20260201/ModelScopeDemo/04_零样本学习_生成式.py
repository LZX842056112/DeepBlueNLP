# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/31 14:28
Create User : 19410
Desc : 底层是生成模型

https://modelscope.cn/models/iic/nlp_mt5_zero-shot-augment_chinese-base
"""
import json
import os
import random

os.environ['XDG_CACHE_HOME'] = r"D:\huggingface"
os.environ['CACHE_HOME'] = r'D:\huggingface'
os.environ['MODELSCOPE_CACHE'] = r'D:\huggingface\modelscope\hub'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch


@torch.no_grad()
def interface01():
    from modelscope.pipelines import pipeline
    from transformers import GenerationConfig

    """
    内部模型就是一个transformer结构的模型，包含一个编码器和一个解码器
        模型训练的时候：
            编码器的输入: 
                文本分类。\n候选标签：交通查询,智能家居。\n文本内容：还有双鸭山到淮阴的汽车票吗13号的
                评价对象抽取：颐和园还是挺不错的，作为皇家园林，有山有水，亭台楼阁，古色古香，见证着历史的变迁。
                评价对象抽取：苏州园林，在建筑中独树一帜，有重大成就的古典园林建筑。苏州园林又称苏州古典园林， 以私家园林为主。
                翻译成英文：如果日本沉没，中国会接收日本难民吗？
            对应的解码器输入以及输出:
                交通查询
                颐和园
                苏州园林
                will China accept Japanese refugees if Japan sinks?

    """

    # 控制生成逻辑的相关配置信息
    generation_config = GenerationConfig(
        # top_k=50,  # 当进行sample采样的时候，仅保留多少个最大置信度的预测token
        # temperature=2.0,  # 温度系数 --> 更改模型的输出置信度
        # top_p=0.9,  # 范围系数，所有置信度大的token预测概率累计值不能超过top_p
        # do_sample=True, num_beams=1  # sample的主要参数
        # num_beams=4, do_sample=False, num_beam_groups=1  # beam search的主要参数
        num_beams=4, do_sample=True, num_beam_groups=1  # beam sample的主要参数
        # num_beams=4, do_sample=False, num_beam_groups=2, diversity_penalty=0.1  # group beam search的主要参数
        # num_beams=1, do_sample=False, penalty_alpha=0.2  # top_k>1, contrastive_search的主要参数
    )

    t2t_generator = pipeline(
        "text2text-generation",
        "iic/nlp_mt5_zero-shot-augment_chinese-base",
        model_revision='master'
    )
    # modelscope.pipelines.nlp.text_generation_pipeline.TextGenerationT5Pipeline
    # modelscope.models.nlp.T5.text2text_generation.T5ForConditionalGeneration
    # modelscope.preprocessors.nlp.text_generation_preprocessor.TextGenerationTransformersPreprocessor
    print(type(t2t_generator))
    print(type(t2t_generator.model))
    print(type(t2t_generator.preprocessor))
    print(t2t_generator.model)

    """
    所有文本分类按照下列格式进行输入:
        文本分类。\n候选标签：class_name1,class_name2,...,class_name_n。\n文本内容：text

        其中:
            class_name1,class_name2,...,class_name_n 就填充当前需要的分类类别名称列表
            text 就填充当前待预测文本
    """

    print("=" * 50)
    print(t2t_generator(
        "文本分类。\n候选标签：天气查询,交通查询,智能家居。\n文本内容：还有双鸭山到淮阴的汽车票吗13号的",
        generation_config=generation_config
    ))
    print("=" * 50)

    print("=" * 50)
    print(t2t_generator(
        "文本分类。\n候选标签：故事,房产,娱乐,文化,游戏,国际,股票,科技,军事,教育。\n文本内容：他们的故事平静而闪光，一代人奠定沉默的基石，让中国走向繁荣。"
    ))
    # {'text': '文化'}
    print("=" * 50)

    print("=" * 50)
    print(t2t_generator(
        "抽取关键词：\n在分析无线Mesh网路由协议所面临挑战的基础上,结合无线Mesh网络的性能要求,以优化链路状态路由(OLSR)协议为原型,采用跨层设计理论,提出了一种基于链路状态良好程度的路由协议LR-OLSR.该协议引入了认知无线网络中的环境感知推理思想,通过时节点负载、链路投递率和链路可用性等信息进行感知,并以此为依据对链路质量进行推理,获得网络中源节点和目的节点对之间各路径状态良好程度的评价,将其作为路由选择的依据,实现对路由的优化选择,提高网络的吞吐量,达到负载均衡.通过与OLSR及其典型改进协议P-OLSR、SC-OLSR的对比仿真结果表明,LR-OLSB能够提高网络中分组的递交率,降低平均端到端时延,在一定程度上达到负载均衡."
    ))
    # {'text': '无线Mesh网,路由协议,环境感知推理'}
    print("=" * 50)

    print("=" * 50)
    print(t2t_generator(
        "为以下的文本生成标题：\n在分析无线Mesh网路由协议所面临挑战的基础上,结合无线Mesh网络的性能要求,以优化链路状态路由(OLSR)协议为原型,采用跨层设计理论,提出了一种基于链路状态良好程度的路由协议LR-OLSR.该协议引入了认知无线网络中的环境感知推理思想,通过时节点负载、链路投递率和链路可用性等信息进行感知,并以此为依据对链路质量进行推理,获得网络中源节点和目的节点对之间各路径状态良好程度的评价,将其作为路由选择的依据,实现对路由的优化选择,提高网络的吞吐量,达到负载均衡.通过与OLSR及其典型改进协议P-OLSR、SC-OLSR的对比仿真结果表明,LR-OLSB能够提高网络中分组的递交率,降低平均端到端时延,在一定程度上达到负载均衡."
    ))
    # {'text': '基于链路状态良好程度的无线Mesh网路由协议'}
    print("=" * 50)

    print("=" * 50)
    print(t2t_generator(
        "为下面的文章生成摘要：\n据统计，今年三季度大中华区共发生58宗IPO交易，融资总额为60亿美元，交易宗数和融资额分别占全球的35%和25%。报告显示，三季度融资额最高的三大证券交易所分别为东京证券交易所、深圳证券交易所和马来西亚证券交易所"))
    # {'text': '大中华区IPO融资额超60亿美元'}
    print("=" * 50)

    print("=" * 50)
    print(t2t_generator("评价对象抽取：颐和园还是挺不错的，作为皇家园林，有山有水，亭台楼阁，古色古香，见证着历史的变迁。"))
    # {'text': '颐和园'}
    print("=" * 50)

    print("=" * 50)
    print(t2t_generator("翻译成英文：如果日本沉没，中国会接收日本难民吗？"))
    # {'text': 'will China accept Japanese refugees if Japan sinks?'}
    print("=" * 50)

    print("=" * 50)
    print(t2t_generator("翻译成日语：如果日本沉没，中国会接收日本难民吗？"))
    # {'text': 'will China accept Japanese refugees if Japan sinks?'}
    print("=" * 50)

    print("=" * 50)
    print(t2t_generator("情感分析：外观漂亮，性能不错，屏幕很好。"))
    # {'text': '积极'}
    print("=" * 50)

    print(t2t_generator(
        "根据给定的段落和答案生成对应的问题。\n段落：跑步后不能马上进食，运动与进食的时间要间隔30分钟以上。看你跑步的量有多大。不管怎么样，跑完步后要慢走一段时间，将呼吸心跳体温调整至正常状态才可进行正常饮食。血液在四肢还没有回流到内脏，不利于消化，加重肠胃的负担。如果口渴可以喝一点少量的水。洗澡的话看你运动量。如果跑步很剧烈，停下来以后，需要让身体恢复正常之后，再洗澡，能达到放松解乏的目的，建议15-20分钟后再洗澡；如果跑步不是很剧烈，只是慢跑，回来之后可以马上洗澡。 \n 答案：30分钟以上"
    ))
    # {'text': '跑步后多久进食'}


def training01():
    """
    直接从网页上copy下来的代码
    :return:
    """
    from modelscope.msdatasets import MsDataset
    from modelscope.metainfo import Trainers
    from modelscope.trainers import build_trainer

    work_dir = r'./workspace/generator/DuReader_robust-QG'

    # DuReader_robust-QG 为示例数据集，用户也可以使用自己的数据集进行训练
    dataset_dict = MsDataset.load('DuReader_robust-QG')

    # 训练数据的输入出均为文本，需要将数据集预处理为输入为 src_txt，输出为 tgt_txt 的格式：
    train_dataset = dataset_dict['train'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'})
    eval_dataset = dataset_dict['validation'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'})
    print(f"训练数据:\n{train_dataset}")
    print(f"验证数据:\n{eval_dataset}")

    num_warmup_steps = 500

    def noam_lambda(current_step: int):
        current_step += 1
        return min(current_step ** (-0.5),
                   current_step * num_warmup_steps ** (-1.5))

    # 可以在代码修改 configuration 的配置
    def cfg_modify_fn(cfg):
        cfg.train.lr_scheduler = {
            'type': 'LambdaLR',
            'lr_lambda': noam_lambda,
            'options': {
                'by_epoch': False
            }
        }
        cfg.train.optimizer = {
            "type": "AdamW",
            "lr": 1e-3,
            "options": {}
        }
        cfg.train.max_epochs = 2  # 15
        cfg.train.dataloader = {
            "batch_size_per_gpu": 4,
            "workers_per_gpu": 0
        }

        # 新增
        # from modelscope.utils.config import ConfigDict
        # if 'evaluation' not in cfg:
        #     cfg['evaluation'] = ConfigDict()
        # cfg.evaluation.dataloader = {
        #     "batch_size_per_gpu": 4,
        #     "workers_per_gpu": 0
        # }
        #
        # _hooks = cfg.train.hooks or []
        # # from modelscope import Tasks
        # # from modelscope.metainfo import Models, Heads, Preprocessors, TaskModels, Trainers, Hooks, Metrics
        # # from modelscope.utils.constant import ModelFile
        # _hooks.extend([
        #     {
        #         'type': 'TextLoggerHook',
        #         'interval': 1
        #     },
        #     {
        #         "type": "CheckpointHook",
        #         "interval": 1
        #     }
        # ])
        # cfg.train.hooks = _hooks
        #
        # from modelscope.metainfo import Metrics
        # cfg.evaluation.metrics = [Metrics.text_gen_metric]

        return cfg

    def data_collator_fn(batch):
        from torch.utils.data import default_collate

        batch = default_collate(batch)

        batch['labels'] = batch['labels'].to(dtype=torch.long)

        return batch

    kwargs = dict(
        model='iic/nlp_mt5_zero-shot-augment_chinese-base',
        model_revision="master",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        work_dir=work_dir,
        cfg_modify_fn=cfg_modify_fn,
        data_collator=data_collator_fn
    )
    trainer = build_trainer(
        name=Trainers.text_generation_trainer, default_args=kwargs
    )
    print(type(trainer))
    trainer.train()


def training02():
    """
    基于问题生成答案
    :return:
    """
    from modelscope.msdatasets import MsDataset
    from modelscope.metainfo import Trainers
    from modelscope.trainers import build_trainer

    work_dir = r'./workspace/generator/DuReader_robust-QG'

    # DuReader_robust-QG 为示例数据集，用户也可以使用自己的数据集进行训练
    dataset_dict = MsDataset.load('DuReader_robust-QG')

    # 训练数据的输入出均为文本，需要将数据集预处理为输入为 src_txt，输出为 tgt_txt 的格式：
    train_dataset = dataset_dict['train'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'})
    eval_dataset = dataset_dict['validation'].remap_columns({'text1': 'src_txt', 'text2': 'tgt_txt'})

    # 对数据进行转换处理
    def reverse_record(record):
        src = record['src_txt']
        answer, text = src.split("[SEP]", maxsplit=1)
        tgt = record['tgt_txt']
        # 数据重新构造一下
        record['src_txt'] = f'{tgt}[SEP]{text}'
        record['tgt_txt'] = answer
        return record

    train_dataset = train_dataset.map(reverse_record)
    eval_dataset = eval_dataset.map(reverse_record)

    train_dataset = train_dataset.filter(lambda t: random.random() < 0.001)
    eval_dataset = eval_dataset.filter(lambda t: random.random() < 0.005)

    print(f"训练数据:\n{train_dataset} - {train_dataset[0]}")
    print(f"验证数据:\n{eval_dataset}- {eval_dataset[0]}")

    num_warmup_steps = 500

    def noam_lambda(current_step: int):
        current_step += 1
        return min(current_step ** (-0.5),
                   current_step * num_warmup_steps ** (-1.5))

    # 可以在代码修改 configuration 的配置
    def cfg_modify_fn(cfg):
        cfg.train.lr_scheduler = {
            'type': 'LambdaLR',
            'lr_lambda': noam_lambda,
            'options': {
                'by_epoch': False
            }
        }
        cfg.train.optimizer = {
            "type": "AdamW",
            "lr": 1e-3,
            "options": {}
        }
        cfg.train.max_epochs = 2  # 15
        cfg.train.dataloader = {
            "batch_size_per_gpu": 4,
            "workers_per_gpu": 0
        }

        # 新增
        # from modelscope.utils.config import ConfigDict
        # if 'evaluation' not in cfg:
        #     cfg['evaluation'] = ConfigDict()
        # cfg.evaluation.dataloader = {
        #     "batch_size_per_gpu": 4,
        #     "workers_per_gpu": 0
        # }
        #
        # _hooks = cfg.train.hooks or []
        # # from modelscope import Tasks
        # # from modelscope.metainfo import Models, Heads, Preprocessors, TaskModels, Trainers, Hooks, Metrics
        # # from modelscope.utils.constant import ModelFile
        # _hooks.extend([
        #     {
        #         'type': 'TextLoggerHook',
        #         'interval': 1
        #     },
        #     {
        #         "type": "CheckpointHook",
        #         "interval": 1
        #     }
        # ])
        # cfg.train.hooks = _hooks
        #
        # from modelscope.metainfo import Metrics
        # cfg.evaluation.metrics = [Metrics.text_gen_metric]

        return cfg

    def data_collator_fn(batch):
        from torch.utils.data import default_collate

        batch = default_collate(batch)

        batch['labels'] = batch['labels'].to(dtype=torch.long)

        return batch

    kwargs = dict(
        model='iic/nlp_mt5_zero-shot-augment_chinese-base',
        model_revision="master",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        work_dir=work_dir,
        cfg_modify_fn=cfg_modify_fn,
        data_collator=data_collator_fn
    )
    trainer = build_trainer(
        name=Trainers.text_generation_trainer, default_args=kwargs
    )
    print(type(trainer))
    trainer.train()


def training03():
    """
    基于自定义的问答对数据训练模型
    :return:
    """
    from modelscope.msdatasets import MsDataset
    from modelscope.metainfo import Trainers
    from modelscope.trainers import build_trainer

    work_dir = r'./workspace/generator/custom-datas'

    # DuReader_robust-QG 为示例数据集，用户也可以使用自己的数据集进行训练
    dataset_dict = MsDataset.load(
        "csv",
        data_files={
            "train": r"./datas/generate_datas/data.csv",
            "validation": r"./datas/generate_datas/data.csv"
        },
        delimiter=","
    )

    # 训练数据的输入出均为文本，需要将数据集预处理为输入为 src_txt，输出为 tgt_txt 的格式：
    train_dataset = dataset_dict['train'].to_hf_dataset()
    eval_dataset = dataset_dict['validation'].to_hf_dataset()

    print(f"训练数据:\n{train_dataset} - \n{train_dataset[0]}")
    print(f"验证数据:\n{eval_dataset}- \n{eval_dataset[0]}")

    num_warmup_steps = 500

    def noam_lambda(current_step: int):
        current_step += 1
        return min(current_step ** (-0.5),
                   current_step * num_warmup_steps ** (-1.5))

    # 可以在代码修改 configuration 的配置
    def cfg_modify_fn(cfg):
        cfg.train.lr_scheduler = {
            'type': 'LambdaLR',
            'lr_lambda': noam_lambda,
            'options': {
                'by_epoch': False
            }
        }
        cfg.train.optimizer = {
            "type": "AdamW",
            "lr": 1e-3,
            "options": {}
        }
        cfg.train.max_epochs = 2  # 15
        cfg.train.dataloader = {
            "batch_size_per_gpu": 2,
            "workers_per_gpu": 0
        }

        # 新增
        from modelscope.utils.config import ConfigDict
        if 'evaluation' not in cfg:
            cfg['evaluation'] = ConfigDict()
        cfg.evaluation.dataloader = {
            "batch_size_per_gpu": 1,
            "workers_per_gpu": 0
        }

        # _hooks = cfg.train.hooks or []
        # # from modelscope import Tasks
        # # from modelscope.metainfo import Models, Heads, Preprocessors, TaskModels, Trainers, Hooks, Metrics
        # # from modelscope.utils.constant import ModelFile
        # _hooks.extend([
        #     {
        #         'type': 'TextLoggerHook',
        #         'interval': 1
        #     },
        #     {
        #         "type": "CheckpointHook",
        #         "interval": 1
        #     }
        # ])
        # cfg.train.hooks = _hooks
        #
        # from modelscope.metainfo import Metrics
        # cfg.evaluation.metrics = [Metrics.text_gen_metric]

        return cfg

    def data_collator_fn(batch):
        from torch.utils.data import default_collate

        batch = default_collate(batch)

        batch['labels'] = batch['labels'].to(dtype=torch.long)

        return batch

    kwargs = dict(
        model='iic/nlp_mt5_zero-shot-augment_chinese-base',
        model_revision="master",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        work_dir=work_dir,
        cfg_modify_fn=cfg_modify_fn,
        data_collator=data_collator_fn
    )
    # modelscope.trainers.nlp.text_generation_trainer.TextGenerationTrainer
    trainer = build_trainer(
        name=Trainers.text_generation_trainer, default_args=kwargs
    )
    print(type(trainer))
    trainer.train()


def training04():
    """
    对联生成
    :return:
    """
    from modelscope.msdatasets import MsDataset
    from modelscope.metainfo import Trainers
    from modelscope.trainers import build_trainer

    work_dir = r'./workspace/generator/couplet'

    dataset_dict = MsDataset.load(
        "text",
        data_files={
            "train": r".\datas\couplet_gen\poetry_min.txt",
            "validation": r".\datas\couplet_gen\poetry_min.txt"
        }
    )

    def parse_record(record):
        text = record['text'].strip()
        src, tgt = text.split("，")
        tgt = tgt[:-1]

        # 基于上联生成下联的结构
        record['src_txt'] = src
        record['tgt_txt'] = tgt
        del record['text']

        return record

    # 训练数据的输入出均为文本，需要将数据集预处理为输入为 src_txt，输出为 tgt_txt 的格式：
    train_dataset = dataset_dict['train'].to_hf_dataset().map(parse_record)
    eval_dataset = dataset_dict['validation'].to_hf_dataset().map(parse_record)

    print(f"训练数据:\n{train_dataset} - \n{train_dataset[0]}")
    print(f"验证数据:\n{eval_dataset}- \n{eval_dataset[0]}")

    num_warmup_steps = 500

    def noam_lambda(current_step: int):
        current_step += 1
        return min(current_step ** (-0.5),
                   current_step * num_warmup_steps ** (-1.5))

    # 可以在代码修改 configuration 的配置
    def cfg_modify_fn(cfg):
        cfg.train.lr_scheduler = {
            'type': 'LambdaLR',
            'lr_lambda': noam_lambda,
            'options': {
                'by_epoch': False
            }
        }
        cfg.train.optimizer = {
            "type": "AdamW",
            "lr": 1e-3,
            "options": {}
        }
        cfg.train.max_epochs = 2  # 15
        cfg.train.dataloader = {
            "batch_size_per_gpu": 2,
            "workers_per_gpu": 0
        }

        # 新增
        from modelscope.utils.config import ConfigDict
        if 'evaluation' not in cfg:
            cfg['evaluation'] = ConfigDict()
        cfg.evaluation.dataloader = {
            "batch_size_per_gpu": 1,
            "workers_per_gpu": 0
        }

        # _hooks = cfg.train.hooks or []
        # # from modelscope import Tasks
        # # from modelscope.metainfo import Models, Heads, Preprocessors, TaskModels, Trainers, Hooks, Metrics
        # # from modelscope.utils.constant import ModelFile
        # _hooks.extend([
        #     {
        #         'type': 'TextLoggerHook',
        #         'interval': 1
        #     },
        #     {
        #         "type": "CheckpointHook",
        #         "interval": 1
        #     }
        # ])
        # cfg.train.hooks = _hooks
        #
        # from modelscope.metainfo import Metrics
        # cfg.evaluation.metrics = [Metrics.text_gen_metric]

        cfg.preprocessor.sequence_length = 15  # 编码器和解码器的输入文本最长长度为15

        return cfg

    def data_collator_fn(batch):
        from torch.utils.data import default_collate

        # 采用默认数据拼接聚合方法 -- 内部是不考虑填充操作的
        batch = default_collate(batch)

        batch['labels'] = batch['labels'].to(dtype=torch.long)

        return batch

    kwargs = dict(
        model='iic/nlp_mt5_zero-shot-augment_chinese-base',
        model_revision="master",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        work_dir=work_dir,
        cfg_modify_fn=cfg_modify_fn,
        data_collator=data_collator_fn
    )
    # modelscope.trainers.nlp.text_generation_trainer.TextGenerationTrainer
    trainer = build_trainer(
        name=Trainers.text_generation_trainer, default_args=kwargs
    )
    print(type(trainer))
    trainer.train()


if __name__ == '__main__':
    # interface01()
    # training01()
    # training02()
    # training03()
    training04()

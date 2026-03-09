# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/25 15:15
Create User : 19410
Desc : 零样本学习：在任意一个新的业务场景中，模型不需要进行任何微调/训练，都可以达到一定的效果

https://modelscope.cn/aggregatedTopic
https://modelscope.cn/topic/9ae88b6a1ffd4de59a9f1948314ebc2b/pub/summary
https://modelscope.cn/models/iic/nlp_structbert_siamese-uninlu_chinese-base

抽取式：指针网络 --> 判断token是否属于最终结果片段的开头或者结尾
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
    """
    主要查看一下各个类的名称
    :return:
    """
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from modelscope.pipelines.nlp.siamese_uie_pipeline import SiameseUiePipeline
    from modelscope.preprocessors.nlp.siamese_uie_preprocessor import SiameseUiePreprocessor
    from modelscope.models.nlp.bert.siamese_uie import SiameseUieModel

    semantic_cls = pipeline(
        Tasks.siamese_uie,
        'damo/nlp_structbert_siamese-uninlu_chinese-base',
        model_revision='master'
    )
    print(semantic_cls)
    print(type(semantic_cls))
    print(type(semantic_cls.preprocessor))
    print(type(semantic_cls.model))
    print(semantic_cls.model)


def interface02():
    """
    推理应用代码
    :return:
    """
    from modelscope import pipeline, Tasks
    from modelscope.pipelines.nlp.siamese_uie_pipeline import SiameseUiePipeline
    from modelscope.models.nlp.bert.siamese_uie import SiameseUieModel
    from modelscope.preprocessors.nlp.siamese_uie_preprocessor import SiameseUiePreprocessor

    model = r'damo/nlp_structbert_siamese-uninlu_chinese-base'
    # model = r"./workspace/nlp_structbert_siamese/output"

    semantic_cls = pipeline(
        Tasks.siamese_uie,
        model=model,
        model_revision='master'
    )

    # """
    # 内部逻辑实际上是 prompt + 指针网络 + 参数共享(相同文本、不同提示词数据之间的共享)
    # eg:
    #     给我找一趟株洲去醴陵的最便宜的汽车票价格 ---> 出发地/目的地/交通工具
    #     执行逻辑：
    #         1. 样本转换为三个样本，分别进行处理
    #             [CLS]给我找一趟株洲去醴陵的最便宜的汽车票价格[SEP][CLS]出发地: [SEP]
    #             [CLS]给我找一趟株洲去醴陵的最便宜的汽车票价格[SEP][CLS]目的地: [SEP]
    #             [CLS]给我找一趟株洲去醴陵的最便宜的汽车票价格[SEP][CLS]交通工具: [SEP]
    #         2. 输入到bert模型中，得到最后一个输出的高阶特征向量，并仅获取第一部分的高阶特征向量
    #         3. 基于提取出来的高阶特征向量，分别进行head分类和tail分类，得到每个token属于实体开头和结尾的置信度
    #         4. 遍历提取实体span
    #         NOTE: 这种训练方式，使用时就相当于通过控制不同的prompt，让相同text输出不同的span
    #               内部就相当于需要模型基于prompt和text之间的语义关系进行特征的提取学习
    #     为了减少计算量:
    #         将1和2进行简化:
    #             a. 先提取原始文本在bert模型中，前6层的特征向量
    #                 [CLS]给我找一趟株洲去醴陵的最便宜的汽车票价格[SEP] --> 6层输出的特征向量 v0
    #             b. 在分别获取prompt对应的前6层特征向量
    #                 [CLS]出发地: [SEP]  -> u1
    #                 [CLS]目的地: [SEP]  -> u2
    #                 [CLS]交通工具: [SEP]  -> u3
    #             c. 将v0和u1、u2、u3进行合并，在输出到bert的后六层，得到最终的交叉特征向量的输出
    #                 (v0,u1) -bert6~12-> z1
    #                 (v0,u2) -bert6~12-> z2
    #                 (v0,u3) -bert6~12-> z3
    # """

    r = semantic_cls(
        # input="相比之下，青岛海牛队和广州松日队的雨中之战虽然也是0∶0，但乏善可陈。",
        input="这个出版社与总社新闻研究所、中国新闻学院合作编撰的国家“九五”重点图书《毛泽东、邓小平、江泽民新闻宣传思想研究》已取得实质性进展。	",
        schema={
            '人名': None,
            '地名': None,
            '机构名': None
        }
    )
    print("=" * 100)
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    r = semantic_cls(
        input='可以通过火车从北京去长沙吗',
        schema={
            '出发地': None,
            '目的地': None,
            '交通工具': None
        }
    )
    print("=" * 100)
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    r = semantic_cls(
        input='可以通过火车从北京去长沙吗',
        schema={
            '出发地': {
                '目的地': None,
                '交通工具': None
            }
        },
        # output_all_prefix=True
    )
    print("=" * 100)
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    r = semantic_cls(
        input='交通查询,天气查询,智能家居|可以通过火车从北京去长沙吗',
        schema={
            '分类': None
        }
    )
    print("=" * 100)
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    r = semantic_cls(
        input='可以通过火车从北京去长沙吗',
        schema={
            '出发地': {
                '目的地': {
                    '交通工具': None
                },
                '交通工具': {
                    '目的地': None
                },
            }
        },
        output_all_prefix=True
    )
    print("=" * 100)
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    r = semantic_cls(
        input='患者因“结肠癌”于2012-12-13在我院于全麻上行右半结肠切15除术，手术过程顺利，术后给予抗感染及营养支持治疗',
        schema={
            '疾病': None,
            '手术类型': None,
            '治疗方式': None
        }
    )
    print("=" * 100)
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    # 命名实体识别 {实体类型: None}
    r = semantic_cls(
        input='1944年毕业于北大的名古屋铁道会长谷口清太郎等人在日本积极筹资，共筹款2.7亿日元，参加捐款的日本企业有69家。',
        schema={
            '人物': None,
            '地理位置': None,
            '组织机构': None
        }
    )
    print("=" * 100)
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    # 关系抽取 {主语实体类型: {关系(宾语实体类型): None}}
    print("=" * 100)
    r = semantic_cls(
        input='在北京冬奥会自由式中，2月8日上午，滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌。2月9日上午，滑雪男子大跳台决赛中日本选手小泉次郎以188.25分获得银牌！',
        schema={
            '人物': {
                '比赛项目(赛事名称)': None,
                '参赛地点(城市)': None,
                '获奖时间(时间)': None,
                '选手国籍(国籍)': None
            }
        }
    )
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    # 事件抽取 {事件类型（事件触发词）: {参数类型: None}}
    # [CLS]7月28日，天津泰达在德比战中以0-1负于天津天海。[SEP][CLS]胜负(事件触发词): [SEP]
    # [CLS]7月28日，天津泰达在德比战中以0-1负于天津天海。[SEP][CLS]胜负(事件触发词): 负, 时间: [SEP]
    # [CLS]7月28日，天津泰达在德比战中以0-1负于天津天海。[SEP][CLS]胜负(事件触发词): 负, 败者: [SEP]
    # [CLS]7月28日，天津泰达在德比战中以0-1负于天津天海。[SEP][CLS]胜负(事件触发词): 负, 胜者: [SEP]
    # [CLS]7月28日，天津泰达在德比战中以0-1负于天津天海。[SEP][CLS]胜负(事件触发词): 负, 赛事名称: [SEP]
    print("=" * 100)
    r = semantic_cls(
        input='7月28日，天津泰达在德比战中以0-1负于天津天海。',
        schema={
            '胜负(事件触发词)': {
                '时间': None,
                '败者': None,
                '胜者': None,
                '赛事名称': None
            }
        }
    )
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    # 属性情感抽取 {属性词: {情感词: None}}
    print("=" * 100)
    r = semantic_cls(
        input='很满意，音质很好，发货速度快，值得购买',
        schema={
            '属性词': {
                '情感词': None,
            }
        }
    )
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    # 允许属性词缺省，#表示缺省
    print("=" * 100)
    r = semantic_cls(
        input='#很满意，音质很好，发货速度快，值得购买',
        schema={
            '属性词': {
                '情感词': None,
            }
        }
    )
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    # 支持情感分类
    # [CLS]#很满意，音质很好，发货速度快，值得购买[SEP][CLS]属性词: [SEP]
    # y_head: 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0
    # y_tail: 0 1 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0
    # [CLS]#很满意，音质很好，发货速度快，值得购买[SEP][CLS]属性词: #, 正向情感(情感词): [SEP]
    # y_head: 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # y_tail: 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # [CLS]#很满意，音质很好，发货速度快，值得购买[SEP][CLS]属性词: #, 负向情感(情感词): [SEP]
    # y_head: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # y_tail: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # [CLS]#很满意，音质很好，发货速度快，值得购买[SEP][CLS]属性词: #, 中性情感(情感词): [SEP]
    # [CLS]#很满意，音质很好，发货速度快，值得购买[SEP][CLS]属性词: 音质, 正向情感(情感词): [SEP]
    # [CLS]#很满意，音质很好，发货速度快，值得购买[SEP][CLS]属性词: 音质, 负向情感(情感词): [SEP]
    # [CLS]#很满意，音质很好，发货速度快，值得购买[SEP][CLS]属性词: 音质, 中性情感(情感词): [SEP]
    # [CLS]#很满意，音质很好，发货速度快，值得购买[SEP][CLS]属性词: 发货速度, 正向情感(情感词): [SEP]
    # [CLS]#很满意，音质很好，发货速度快，值得购买[SEP][CLS]属性词: 发货速度, 负向情感(情感词): [SEP]
    # [CLS]#很满意，音质很好，发货速度快，值得购买[SEP][CLS]属性词: 发货速度, 中性情感(情感词): [SEP]
    print("=" * 100)
    r = semantic_cls(
        input='#很满意，音质很好，发货速度快，值得购买',
        schema={
            '属性词': {
                "正向情感(情感词)": None,
                "负向情感(情感词)": None,
                "中性情感(情感词)": None
            }
        }
    )
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    # 指代消解，判断选项通过英文逗号“,”隔开，拼接在输入文本前面并用“｜”分隔
    print("=" * 100)
    r = semantic_cls(
        input='是的,不是|哥哥点了点头。“我这几年苦哇……现在玲玲也大一点了，所以……”他望着妹妹（候选词），脸上显出一副要求她(代词)谅解的表情。',
        schema={
            '在下面的描述中，代词“她”指代的是“妹妹”吗？': None
        }
    )
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    # 情感分类，情感标签通过英文逗号“,”隔开，拼接在输入文本前面并用“｜”分隔；同时也支持情绪分类任务，换成相应情绪标签即可，e.g. "无情绪,积极,愤怒,悲伤,恐惧,惊奇"
    print("=" * 100)
    r = semantic_cls(
        input='正向,负向|有点看不下去了，看作者介绍就觉得挺矫情了，文字也弱了点。后来才发现 大家对这本书评价都很低。亏了。',
        schema={
            '情感分类': None
        }
    )
    print("=" * 100)
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    # 文本分类，文本标签通过英文逗号“,”隔开，拼接在输入文本前面并用“｜”分隔
    print("=" * 100)
    r = semantic_cls(
        input='民生故事,文化,娱乐,体育,财经,房产,汽车,教育,科技,军事,旅游,国际,证券股票,农业三农,电竞游戏|学校召开2018届升学及出国深造毕业生座谈会就业指导',
        schema={
            '分类': None
        }
    )
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    print("=" * 100)
    r = semantic_cls(
        input='交通查询,智能家居|还有双鸭山到淮阴的汽车票吗13号的',
        schema={
            '分类': None
        }
    )
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    # 文本匹配，文本相似度标签通过英文逗号“,”隔开，拼接在输入文本前面并用“｜”分隔；输入文本由两段文本组成，并用“&”隔开
    print("=" * 100)
    r = semantic_cls(
        input='相似,不相似|摄像头区域遮挡屏幕&通话遮挡屏幕黑屏正常',
        schema={
            '文本匹配': None
        }
    )
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    # 文本匹配也可以用下面这种方式组织，判断选项通过英文逗号“,”隔开，拼接在输入文本前面并用“｜”分隔；输入文本由两段文本组成，并分别用“句子1”和“句子2”区分
    print("=" * 100)
    r = semantic_cls(
        input='是的,不是|句子1：摄像头区域遮挡屏幕；句子2：通话遮挡屏幕黑屏正常',
        schema={
            '下面两句话的意思是否相同': None
        }
    )
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    # 自然语言推理，文本关系标签通过英文逗号“,”隔开，拼接在输入文本前面并用“｜”分隔；输入文本由两段文本组成，并分别用“段落1”和“段落2”区分
    print("=" * 100)
    r = semantic_cls(
        input='蕴含,矛盾,中立|段落1：是,但是你比如说像现在这种情况,是不是就是说咱们离它就绝对人类是再也没有任何可能性了；段落2：我对人类可能性有所思考',
        schema={
            '段落2和段落1的关系是：': None
        }
    )
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    # 选择类阅读理解，选项通过英文逗号“,”隔开，拼接在输入文本前面并用“｜”分隔
    print("=" * 100)
    r = semantic_cls(
        input='飞机票太贵,时间来不及,坐飞机头晕,飞机票太便宜|A：最近飞机票打折挺多的，你还是坐飞机去吧。B：反正又不是时间来不及，飞机再便宜我也不坐，我一听坐飞机就头晕。',
        schema={
            'B为什么不坐飞机?': None
        }
    )
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    # 抽取式阅读理解，选项通过英文逗号“,”隔开，拼接在输入文本前面并用“｜”分隔
    print("=" * 100)
    r = semantic_cls(
        input='A：最近飞机票打折挺多的，你还是坐飞机去吧。B：反正又不是时间来不及，飞机再便宜我也不坐，我一听坐飞机就头晕。',
        schema={
            'B为什么不坐飞机?': None
        }
    )
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    # 抽取类阅读理解
    print("=" * 100)
    r = semantic_cls(
        # input='大莱龙铁路位于山东省北部环渤海地区，西起位于益羊铁路的潍坊大家洼车站，向东经海化、寿光、寒亭、昌邑、平度、莱州、招远、终到龙口，连接山东半岛羊角沟、潍坊、莱州、龙口四个港口，全长175公里，工程建设概算总投资11.42亿元。铁路西与德大铁路、黄大铁路在大家洼站接轨，东与龙烟铁路相连。大莱龙铁路于1997年11月批复立项，2002年12月28日全线铺通，2005年6月建成试运营，是横贯山东省北部的铁路干线德龙烟铁路的重要组成部分，构成山东省北部沿海通道，并成为环渤海铁路网的南部干线。铁路沿线设有大家洼站、寒亭站、昌邑北站、海天站、平度北站、沙河站、莱州站、朱桥站、招远站、龙口西站、龙口北站、龙口港站。大莱龙铁路官方网站',
        input="""
                上海虹桥国际机场（Shanghai Hongqiao International Airport，IATA：SHA，ICAO：ZSSS），位于中国上海市长宁区和闵行区交界处，距市中心13千米，为4E级民用国际机场，是中国三大门户复合枢纽之一、 [1]国际定期航班机场、对外开放的一类航空口岸和国际航班备降机场。 [2]
        上海虹桥国际机场始建于1921年，于1950年重建；1971年由军民合用改为民航专用；2010年启用2号航站楼及第二跑道；2014年底启动1号航站楼改造及东交通中心工程。 [3-4]
        据2017年9月综合信息显示，上海虹桥国际机场建筑面积51万平方米；航站楼面积44.46万平方米， [5]拥有跑道两条，分别长3400米、3300米； [6]停机坪约48.6万平米，共有89个机位。 [2] [6]
        2022年，上海虹桥国际机场旅客吞吐量1471.16万人次，同比下降55.7%；货邮吞吐量18.45万吨，同比下降51.9%；起降架次12.27万架次，同比下降47.0%，分别位居中国第7位、第15位、第14位 [51]。
        2024年6月14日，东航C919机型将开启第四条商业定期航线——上海虹桥往返广州白云。 [57]7月1日起，在上海虹桥机场运营的国际、港澳台航班截载时间全部缩短至起飞前45分钟以内。
                """,
        schema={
            # '上海虹桥什么时候创建？': None
            '上海虹桥国际机场有多少个停机位？': None
            # '上海虹桥国际机场2022年货邮吞吐量相比于2021年降低了多少？': None
            # '上海虹桥机场现在的航班截止时间是多少分钟？': None
            # '大莱龙铁路位于哪里？': None
        }
    )
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)


def training01():
    import os
    import json
    from modelscope.trainers import build_trainer
    from modelscope.msdatasets import MsDataset
    from modelscope.utils.hub import read_config
    from modelscope.metainfo import Metrics
    from modelscope.utils.constant import DownloadMode

    model_id = 'damo/nlp_structbert_siamese-uninlu_chinese-base'

    WORK_DIR = 'workspace/nlp_structbert_siamese-uninlu_chinese/zero_sample/people_daily_ner_1998_tiny'

    # 需要修改 modelscope/msdatasets/utils/hf_datasets_util.py 里面的 _dataset_info 方法的入参，增加一个**kwargs入参即可
    train_dataset = MsDataset.load(
        'damo/people_daily_ner_1998_tiny',
        namespace='damo', split='train',
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        trust_remote_code=True
    )
    eval_dataset = MsDataset.load(
        'damo/people_daily_ner_1998_tiny',
        namespace='damo', split='validation',
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
        trust_remote_code=True
    )
    # eval_dataset 是datasets.arrow_dataset.Dataset库里面的DS对象
    eval_dataset = eval_dataset.filter(lambda t: random.random() < 0.1)
    train_dataset = train_dataset.filter(lambda t: random.random() < 0.1)
    print(f"训练数据:\n{train_dataset} - \n{train_dataset[0]}")
    print(f"评估数据:\n{eval_dataset} - \n{eval_dataset[0]}")

    max_epochs = 3

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
                "type": "IterTimerHook"
            },
            {
                'type': 'TextLoggerHook',
                'interval': 1
            },
            {
                "type": "CheckpointHook",
                "interval": 2
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
        cfg.train.optimizer.lr = 3e-5
        cfg.train.lr_scheduler.step_size = 1
        cfg.train.lr_scheduler.options.by_epoch = True
        return cfg

    kwargs = dict(
        model=model_id,
        model_revision='master',
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_epochs=max_epochs,
        work_dir=WORK_DIR,
        cfg_modify_fn=cfg_modify_fn
    )

    # modelscope.trainers.nlp.siamese_uie_trainer.SiameseUIETrainer
    trainer = build_trainer('siamese-uie-trainer', default_args=kwargs)
    print(type(trainer))

    print('===============================================================')
    print('pre-trained model loaded, training started:')
    print('===============================================================')

    """
        如果出现需要安装distutils库，并且当前python版本为3.12+的时候，那么可能需要修改modelscope的源码：
            1. 修改 modelscope/utils/import_utils.py 332行的 SYSTEM_PACKAGE的值
                #SYSTEM_PACKAGE = set(['os', 'sys', 'typing', 'distutils'])
                SYSTEM_PACKAGE = set(['os', 'sys', 'typing'])
            2. 修改 modelscope/utils/ast_index_file.py 的内容，删除requirements中的distutils的依赖
                NOTE: 特别注意 modelscope.trainers.trainer 这个对应的依赖
            3. 修改 modelscope/trainers/trainer.py 内容
                a. 注释第六行的import内容
                    from distutils.version import LooseVersion
                b. 替换 1200行左右 的LooseVersion的使用代码
                    # noinspection PyBroadException
                    def _is_new_torch():
                        try:
                            from packaging import version
            
                            _v = version.parse(torch.__version__)
                            if (_v.major, _v.minor, _v.micro) >= (1, 7, 0):
                                return True
                        except:
                            pass
                        return False
            
                    # if LooseVersion(torch.__version__) >= LooseVersion('1.7.0'):
                    if _is_new_torch():
                        kwargs['persistent_workers'] = persistent_workers

    """
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


def interface03():
    """
    推理应用代码 使用自己训练的模型
    :return:
    """
    from modelscope import pipeline, Tasks
    from modelscope.pipelines.nlp.siamese_uie_pipeline import SiameseUiePipeline
    from modelscope.models.nlp.bert.siamese_uie import SiameseUieModel
    from modelscope.preprocessors.nlp.siamese_uie_preprocessor import SiameseUiePreprocessor

    model = r'damo/nlp_structbert_siamese-uninlu_chinese-base'
    model = r"./workspace/nlp_structbert_siamese-uninlu_chinese/zero_sample/people_daily_ner_1998_tiny/output_best"
    if os.path.exists(model):
        model = os.path.abspath(model)

    semantic_cls = pipeline(
        Tasks.siamese_uie,
        model=model,
        model_revision='master'
    )

    # """
    # 内部逻辑实际上是 prompt + 指针网络 + 参数共享(相同文本、不同提示词数据之间的共享)
    # eg:
    #     给我找一趟株洲去醴陵的最便宜的汽车票价格 ---> 出发地/目的地/交通工具
    #     执行逻辑：
    #         1. 样本转换为三个样本，分别进行处理
    #             [CLS]给我找一趟株洲去醴陵的最便宜的汽车票价格[SEP][CLS]出发地: [SEP]
    #             [CLS]给我找一趟株洲去醴陵的最便宜的汽车票价格[SEP][CLS]目的地: [SEP]
    #             [CLS]给我找一趟株洲去醴陵的最便宜的汽车票价格[SEP][CLS]交通工具: [SEP]
    #         2. 输入到bert模型中，得到最后一个输出的高阶特征向量，并仅获取第一部分的高阶特征向量
    #         3. 基于提取出来的高阶特征向量，分别进行head分类和tail分类，得到每个token属于实体开头和结尾的置信度
    #         4. 遍历提取实体span
    #         NOTE: 这种训练方式，使用时就相当于通过控制不同的prompt，让相同text输出不同的span
    #               内部就相当于需要模型基于prompt和text之间的语义关系进行特征的提取学习
    #     为了减少计算量:
    #         将1和2进行简化:
    #             a. 先提取原始文本在bert模型中，前6层的特征向量
    #                 [CLS]给我找一趟株洲去醴陵的最便宜的汽车票价格[SEP] --> 6层输出的特征向量 v0
    #             b. 在分别获取prompt对应的前6层特征向量
    #                 [CLS]出发地: [SEP]  -> u1
    #                 [CLS]目的地: [SEP]  -> u2
    #                 [CLS]交通工具: [SEP]  -> u3
    #             c. 将v0和u1、u2、u3进行合并，在输出到bert的后六层，得到最终的交叉特征向量的输出
    #                 (v0,u1) -bert6~12-> z1
    #                 (v0,u2) -bert6~12-> z2
    #                 (v0,u3) -bert6~12-> z3
    # """

    r = semantic_cls(
        # input="相比之下，青岛海牛队和广州松日队的雨中之战虽然也是0∶0，但乏善可陈。",
        input="这个出版社与总社新闻研究所、中国新闻学院合作编撰的国家“九五”重点图书《毛泽东、邓小平、江泽民新闻宣传思想研究》已取得实质性进展。	",
        schema={
            '人名': None,
            '地名': None,
            '机构名': None
        }
    )
    print("=" * 100)
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)

    r = semantic_cls(
        input='可以通过火车从北京去长沙吗',
        schema={
            '出发地': None,
            '目的地': None,
            '交通工具': None
        }
    )
    print("=" * 100)
    print(json.dumps(r, ensure_ascii=False))
    print("=" * 100)


def training02():
    import os
    import json
    from modelscope.trainers import build_trainer
    from modelscope.msdatasets import MsDataset
    from modelscope.utils.hub import read_config
    from modelscope.metainfo import Metrics
    from modelscope.utils.constant import DownloadMode

    model_id = 'damo/nlp_structbert_siamese-uninlu_chinese-base'

    WORK_DIR = 'workspace/nlp_structbert_siamese-uninlu_chinese/zero_sample/custom'

    # 需要修改 modelscope/msdatasets/utils/hf_datasets_util.py 里面的 _dataset_info 方法的入参，增加一个**kwargs入参即可
    train_dataset = MsDataset.load(
        'csv',
        data_files=[
            "./datas/uie_data/data.csv"
        ],
        delimiter=","
    ).to_hf_dataset()
    eval_dataset = MsDataset.load(
        'csv',
        data_files=[
            "./datas/uie_data/data.csv"
        ],
        delimiter=","
    ).to_hf_dataset()
    print(f"训练数据:\n{train_dataset} - \n{train_dataset[0]}")
    print(f"评估数据:\n{eval_dataset} - \n{eval_dataset[0]}")

    max_epochs = 3

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
                "type": "IterTimerHook"
            },
            {
                'type': 'TextLoggerHook',
                'interval': 1
            },
            {
                "type": "CheckpointHook",
                "interval": 2
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
        cfg.train.optimizer.lr = 3e-5
        cfg.train.lr_scheduler.step_size = 1
        cfg.train.lr_scheduler.options.by_epoch = True
        return cfg

    kwargs = dict(
        model=model_id,
        model_revision='master',
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_epochs=max_epochs,
        work_dir=WORK_DIR,
        cfg_modify_fn=cfg_modify_fn
    )

    # modelscope.trainers.nlp.siamese_uie_trainer.SiameseUIETrainer
    trainer = build_trainer('siamese-uie-trainer', default_args=kwargs)
    print(type(trainer))

    print('===============================================================')
    print('pre-trained model loaded, training started:')
    print('===============================================================')

    """
        如果出现需要安装distutils库，并且当前python版本为3.12+的时候，那么可能需要修改modelscope的源码：
            1. 修改 modelscope/utils/import_utils.py 332行的 SYSTEM_PACKAGE的值
                #SYSTEM_PACKAGE = set(['os', 'sys', 'typing', 'distutils'])
                SYSTEM_PACKAGE = set(['os', 'sys', 'typing'])
            2. 修改 modelscope/utils/ast_index_file.py 的内容，删除requirements中的distutils的依赖
                NOTE: 特别注意 modelscope.trainers.trainer 这个对应的依赖
            3. 修改 modelscope/trainers/trainer.py 内容
                a. 注释第六行的import内容
                    from distutils.version import LooseVersion
                b. 替换 1200行左右 的LooseVersion的使用代码
                    # noinspection PyBroadException
                    def _is_new_torch():
                        try:
                            from packaging import version

                            _v = version.parse(torch.__version__)
                            if (_v.major, _v.minor, _v.micro) >= (1, 7, 0):
                                return True
                        except:
                            pass
                        return False

                    # if LooseVersion(torch.__version__) >= LooseVersion('1.7.0'):
                    if _is_new_torch():
                        kwargs['persistent_workers'] = persistent_workers

    """
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
    # interface02()
    # training01()
    # interface03()
    training02()

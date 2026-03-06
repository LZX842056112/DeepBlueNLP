#! -*- coding:utf-8 -*-
# bert+crf用来做实体识别
# 数据集：http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# [valid_f1]  token_level: 97.06； entity_level: 95.90
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from bert4torch.callbacks import Callback
from bert4torch.snippets import sequence_padding, ListDataset, seed_everything
from bert4torch.layers import CRF
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from tqdm import tqdm

maxlen = 256
batch_size = 16
# 定义 BIO 标签体系（共 13 个类别标签）
categories = [
    'O', 'B-实验室检验', 'I-实验室检验', 'B-影像检查', 'I-影像检查', 'B-手术', 'I-手术',
    'B-疾病和诊断', 'I-疾病和诊断', 'B-药物', 'I-药物', 'B-解剖部位', 'I-解剖部位'
]
# 构建 ID 和 标签 之间的双向映射字典
categories_id2label = {i: k for i, k in enumerate(categories)}
categories_label2id = {k: i for i, k in enumerate(categories)}

# BERT base
model_dir = r"D:\cache\huggingface\hub\models--bert-base-chinese\snapshots\8f23c25b06e129b6c986331a13d8d025a92cf0ea"
config_path = f'{model_dir}/bert4torch_config.json'
checkpoint_path = f'{model_dir}/pytorch_model.bin'
dict_path = f'{model_dir}/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 固定seed
seed_everything(42)


# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        D = []

        with open(filename, "r", encoding="utf-8") as reader:
            for line in reader:  # 遍历文件中的每一行数据
                line = line.strip()  # 前后空格及不可见字符去除
                obj = json.loads(line)  # json字符串转换为obj对象(字典)

                # 获取得到当前数据的原始文本
                text = obj['originalText']
                d = [text]

                # 遍历句子中的所有标注实体
                # 对应的token 类别id
                for entity in obj['entities']:
                    label_type = entity['label_type']  # 实体类别名称
                    start_pos = entity['start_pos']  # 实体在token中起始位置，包含
                    end_pos = entity['end_pos']  # 实体在token中结束位置，不包含
                    # 将实体坐标加入列表。注意：这里将 end_pos - 1 转为了闭区间 [start, end]
                    d.append([start_pos, end_pos - 1, label_type])
                # d 的结构类似: ["感冒吃感冒灵", [0, 1, "疾病和诊断"], [3, 5, "药物"]]
                D.append(d)
        return D


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for d in batch:
        # 对文本进行切词（会自动在首尾加 [CLS] 和 [SEP]）
        tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
        # 【极其重要】映射对齐机制！
        # 因为 BERT 可能会把一些生僻字变成 [UNK]，或者把 "英文" 切分成 "英" "##文"
        # 导致原始 text 的字符索引，跟 token 的索引对不上。
        # rematch 会建立 原文字符 到 token 的对应关系
        mapping = tokenizer.rematch(d[0], tokens)
        # 构建从原始字符位置到 token 位置的映射字典
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        # 将 token 转为对应的数字 ID
        token_ids = tokenizer.tokens_to_ids(tokens)
        # 初始化全 0 的标签数组（0 对应 'O'）
        labels = np.zeros(len(token_ids))
        # 遍历该句子下的所有实体: start, end, label
        for start, end, label in d[1:]:
            # 只有当原始坐标在映射表中存在时，才进行标签赋值
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]  # 转换为 token 中的起始索引
                end = end_mapping[end]  # 转换为 token 中的结束索引
                # 打上 B 标签 (Begin)
                labels[start] = categories_label2id['B-' + label]
                # 打上 I 标签 (Inside)。注意切片是左闭右开，所以 end 要 +1
                labels[start + 1:end + 1] = categories_label2id['I-' + label]
        batch_token_ids.append(token_ids)
        batch_labels.append(labels)
    # sequence_padding 统一用 0 补齐 batch 内句子长度
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(sequence_padding(batch_labels), dtype=torch.long, device=device)
    return batch_token_ids, batch_labels


# 转换数据集
# noinspection PyTypeChecker
train_dataloader = DataLoader(
    MyDataset(r"../datas/medical/min_training.txt"),
    batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
# noinspection PyTypeChecker
valid_dataloader = DataLoader(
    MyDataset(r"../datas/medical/min_val.txt"),
    batch_size=batch_size, collate_fn=collate_fn
)


# 定义 BERT+CRF 模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        # 1. 底层 BERT 用于提取字向量特征
        self.bert = build_transformer_model(
            config_path=config_path, checkpoint_path=checkpoint_path, segment_vocab_size=0
        )
        # 2. 线性分类头，将 768 维降维到 13 维（标签数），得出 Emission 分数
        self.fc = nn.Linear(768, len(categories))  # 包含首尾

        # CRF ---> 需要在全连接(Softmax)的基础上额外的训练一个标签-标签的序列转换的置信度矩阵(转移概率矩阵/状态转移概率矩阵/类别标签转移概率矩阵)，
        # 也就是上一个时刻是A的时候，当前时刻是各个类别标签的置信度值
        # 最终当前时刻属于类别A的置信度是由两部分构成的：全连接(Softmax)输出属于类别A的置信度 + CRF中学习得到的上一个时刻类别X到当前时刻类别A的置信度
        self.crf = CRF(len(categories))

    def forward(self, token_ids):
        sequence_output = self.bert([token_ids])  # 得到bert的输出特征向量 [btz, seq_len, hdsz]
        emission_score = self.fc(sequence_output)  # 得到每个token属于各个类别的置信度 [btz, seq_len, tag_size]
        # mask 掉 padding 部分（token_id > 0 的才是有效字符）
        attention_mask = token_ids.gt(0).long()
        return emission_score, attention_mask

    def predict(self, token_ids):
        self.eval()
        with torch.no_grad():
            # 得到模型预测各个token属于各个类别的置信度信息以及mask矩阵信息
            emission_score, attention_mask = self.forward(token_ids)
            # 结合crf的模型参数，得到最终各个token对应的预测类别标签
            best_path = self.crf.decode(emission_score, attention_mask)  # [btz, seq_len]
            # best_path = torch.argmax(emission_score, dim=-1)
        return best_path


model = Model().to(device)


# 定义 CRF 的专属损失函数（底层计算的是真实路径得分与其他所有可能路径得分的对数似然）
class Loss(nn.Module):
    def forward(self, outputs, labels):
        # outputs 解包出 emission_score 和 attention_mask
        return model.crf(*outputs, labels)


# 占位符评估指标（仅监控 token 级别准确率，实际意义不大，真实指标看 evaluate 函数）
def acc(y_pred, y_true):
    y_pred = y_pred[0]
    y_pred = torch.argmax(y_pred, dim=-1)
    acc = torch.sum(y_pred.eq(y_true)).item() / y_true.numel()
    return {'acc': acc}


# 支持多种自定义metrics = ['accuracy', acc, {acc: acc}]均可
model.compile(loss=Loss(), optimizer=optim.Adam(model.parameters(), lr=2e-5), metrics=acc)


def evaluate(data):
    """
    计算 Token级别 和 Entity级别 的指标。
    Entity级别更难：必须实体的边界(start, end)和类别(type)完全一致才算对。
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10  # Token 级别（分子分母防止除零）
    X2, Y2, Z2 = 1e-10, 1e-10, 1e-10  # Entity 级别
    for token_ids, label in tqdm(data):
        scores = model.predict(token_ids)  # 预测路径 [btz, seq_len]
        attention_mask = label.gt(0)  # 去除 padding 部分的影响

        # --- token 粒度统计 ---
        X += (scores.eq(label) * attention_mask).sum().item()  # 预测正确个数
        Y += scores.gt(0).sum().item()  # 预测出的实体Token总数
        Z += label.gt(0).sum().item()  # 真实的实体Token总数

        # --- entity 粒度统计 ---
        entity_pred = trans_entity2tuple(scores)  # 将预测标签解析为实体集合
        entity_true = trans_entity2tuple(label)  # 将真实标签解析为实体集合
        X2 += len(entity_pred.intersection(entity_true))  # 交集 = 预测完全正确的实体数
        Y2 += len(entity_pred)  # 预测出的所有实体数
        Z2 += len(entity_true)  # 真实存在的所有实体数

    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    f2, precision2, recall2 = 2 * X2 / (Y2 + Z2), X2 / Y2, X2 / Z2
    return f1, precision, recall, f2, precision2, recall2


def trans_entity2tuple(scores):
    '''把tensor转为(样本id, start, end, 实体类型)的tuple用于计算指标
    '''
    batch_entity_ids = set()
    for i, one_samp in enumerate(scores):
        entity_ids = []
        for j, item in enumerate(one_samp):
            flag_tag = categories_id2label[item.item()]
            if flag_tag.startswith('B-'):  # B
                # 遇到 B，开启一个新实体: [样本索引, 起始位, 当前位(占位), 类型]
                entity_ids.append([i, j, j, flag_tag[2:]])
            elif len(entity_ids) == 0:
                # 没有B开头，直接遇到I，说明预测错了，丢弃
                continue
            elif (len(entity_ids[-1]) > 0) and flag_tag.startswith('I-') and (flag_tag[2:] == entity_ids[-1][-1]):  # I
                # 遇到合法的 I，更新上一个实体的结束位置
                entity_ids[-1][-2] = j
            elif len(entity_ids[-1]) > 0:
                # 遇到 O 或者不合法的 I，闭合实体
                entity_ids.append([])
        # 将有效实体转为 tuple 加入集合
        for i in entity_ids:
            if i:
                batch_entity_ids.add(tuple(i))
    return batch_entity_ids


class Evaluator(Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        # 触发验证集评估
        f1, precision, recall, f2, precision2, recall2 = evaluate(valid_dataloader)
        # 我们通常以 Entity-level 的 F1 分数 (f2) 为核心判断标准
        if f2 > self.best_val_f1:
            self.best_val_f1 = f2
            # 开启此处可以将效果最好的模型保存下来
            # model.save_weights('best_model.pt')
        print(f'[val-token  level] f1: {f1:.5f}, p: {precision:.5f} r: {recall:.5f}')
        print(
            f'[val-entity level] f1: {f2:.5f}, p: {precision2:.5f} r: {recall2:.5f} best_f1: {self.best_val_f1:.5f}\n')


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=20, steps_per_epoch=None, callbacks=[evaluator])
else:
    model.load_weights('best_model.pt')

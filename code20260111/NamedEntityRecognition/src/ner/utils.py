# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/10 15:44
Create User : 19410
Desc : xxx
"""
import json
import os
from typing import List

import numpy as np


def save_json(file, obj):
    """
    将obj对象以json格式的形式保存到对应的磁盘路径
    :param file: 对应文件保存路径
    :param obj: 对应待保存的对象
    :return:
    """
    # 创建输出文件夹
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "w", encoding="utf-8") as writer:
        # noinspection PyTypeChecker
        json.dump(obj, writer, indent=2, ensure_ascii=False)


def load_json(file):
    with open(file, "r", encoding="utf-8") as reader:
        return json.load(reader)


def trans_entity2tuple(label_ids, label_id2names, label_pad_id=-100, offset=0):
    """
    转换对应的标签id到对应的实体片段
    :param label_ids: [bs,t] numpy ndarray bs个样本，每个样本t个token对应的实际类别id
    :param label_id2names: dict[int,str] 类别id到类别名称的映射，并且针对实体来讲，类别名称格式一定为: B-XXX、M-XXX、S-XXX、E-XXX, 类别0属于非实体
    :param label_pad_id：数据中的填充类别id，默认为-100
    :param offset: 样本偏移值
    :return:
    """
    bs, _ = label_ids.shape
    entities = []
    for i in range(bs):  # 每个样本每个样本进行处理
        label_ids_pre_i = label_ids[i]  # 当前样本 [t]

        # (start_pos, end_pos, label_type)
        start_pos, end_pos, label_type = None, None, None
        for _t, _label_id in enumerate(label_ids_pre_i):
            if _label_id == label_pad_id:
                break  # 填充id之后的所有数据不需要考虑

            _label_name = label_id2names[_label_id]  # 当前时刻对应的预测类别名称
            if _label_name.startswith("B-"):
                # 当前token属于实体的开头
                start_pos = _t  # 记录当前实体的开始
                label_type = _label_name[2:]  # 记录当前实体的类别
            elif _label_name.startswith("M-"):
                # 当前token属于实体的中间
                if _label_name[2:] != label_type:
                    # 如果当前实体类别和前面序列的实体类别不匹配，那么表述不是同一个实体，直接重置
                    start_pos, end_pos, label_type = None, None, None
            elif _label_name.startswith("E-"):
                # 当前token属于实体的结尾
                if _label_name[2:] != label_type:
                    # 如果当前实体类别和前面序列的实体类别不匹配，那么表述不是同一个实体，直接重置
                    start_pos, end_pos, label_type = None, None, None
                if start_pos is not None:
                    end_pos = _t + 1
                    entity = (i + offset, start_pos, end_pos, label_type)
                    entities.append(entity)
                start_pos, end_pos, label_type = None, None, None
            elif _label_name.startswith("S-"):
                # 当前token独立属于实体
                entity = (i + offset, _t, _t + 1, _label_name[2:])
                entities.append(entity)
                start_pos, end_pos, label_type = None, None, None
            else:
                # 当前token不属于实体
                start_pos, end_pos, label_type = None, None, None
    return entities


def extract_entities(text: str, pred_entities, sub_text_lengths: List[int], append_special_token=False):
    if append_special_token:
        sub_text_lengths = np.asarray(sub_text_lengths) - 2
        offset = [-1] + list(sub_text_lengths)
    else:
        offset = [0] + list(sub_text_lengths)
    offset_pos_cumsum = np.cumsum(offset)

    final_entities = []
    for sample_id, start_pos, end_pos, label_type in pred_entities:
        offset_pos = offset_pos_cumsum[sample_id]  # 偏移的索引
        sp = int(start_pos + offset_pos)
        ep = int(end_pos + offset_pos)
        final_entities.append({
            "label_type": label_type,
            "start_pos": sp,
            "end_pos": ep,
            'entity': text[sp:ep]
        })
    return final_entities

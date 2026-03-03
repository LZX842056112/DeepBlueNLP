# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/10 15:44
Create User : 19410
Desc : xxx
"""
import json
import os


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

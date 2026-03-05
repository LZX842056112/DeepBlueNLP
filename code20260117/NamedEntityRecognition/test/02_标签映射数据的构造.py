# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/10 15:26
Create User : 19410
Desc : xxx
"""
import json
import os

from ner.utils import save_json


def extract_labels_per_file(file):
    labels = set()
    with open(file, "r", encoding="utf-8") as reader:
        for line in reader:  # 遍历文件中的每一行数据
            line = line.strip()  # 前后空格及不可见字符去除
            obj = json.loads(line)  # json字符串转换为obj对象(字典)
            for entity in obj['entities']:
                label_type = entity['label_type']
                labels.add(label_type)
    return labels


def extract_labels(in_files, out_dir):
    """
    提取 in_files 里面的实体信息，并将结果保存到out_dir路径下
    :param in_files:
    :param out_dir:
    :return:
    """
    if isinstance(in_files, str):
        in_files = [in_files]
    labels = set()
    for in_file in in_files:
        labels.update(extract_labels_per_file(in_file))
    labels = sorted(list(labels))
    print(f"所有的标签列表为:{labels}")

    # 2. 构建成一个字典mapping
    categories = {
        'Other': 0,  # 不属于实体
    }
    for label in labels:
        for prefix in ['B', 'M', 'E', 'S']:
            categories[f'{prefix}-{label}'] = len(categories)
    print(f"映射mapping信息:{categories}")

    # 3. 输出文件夹
    os.makedirs(out_dir, exist_ok=True)
    save_json(os.path.join(out_dir, "label2id.json"), categories)


if __name__ == '__main__':
    # print(extract_labels(
    #     in_files="./datas/medical/training.txt",
    #     out_dir="./datas/medical"
    # ))
    # print(extract_labels(
    #     in_files="./datas/china-people-daily-ner-corpus/ner/min_training.txt",
    #     out_dir="./datas/china-people-daily-ner-corpus/ner"
    # ))
    print(extract_labels(
        in_files="./datas/travel_query/training.txt",
        out_dir="./datas/travel_query"
    ))

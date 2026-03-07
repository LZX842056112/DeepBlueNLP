# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/8/30 14:35
Create User : 19410
Desc : xxx
"""
import os.path
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from toolkit.pre_load import pre_load_thu
from toolkit.NER import get_NE


def tt_ner(text: str):
    thu1 = pre_load_thu  # 提前加载好了
    key = text.strip()
    TagList = thu1.cut(key, text=False)
    NE_List = get_NE(key)  # 获取实体列表
    print(NE_List)
    # print(TagList)


if __name__ == '__main__':
    tt_ner("[药理作用] 诊断试剂 人体内不含菊糖，静注后，不被机体分解、结合、利用和破坏，经肾小球滤过，通过测定血中和尿中的菊糖含量，可以准确计算肾小球的滤过率。菊糖广泛存在于植物组织中,约有3.6万种植物中含有菊糖,尤其是菊芋、菊苣块根中含有丰富的菊糖[6,8]。菊芋(Jerusalem artichoke)又名洋姜,多年生草本植物,在我国栽种广泛,其适应性广、耐贫瘠、产量高、易种植,一般亩产菊芋块茎为2 000～4 000 kg,菊芋块茎除水分外,还含有15%～20%的菊糖,是加工生产菊糖及其制品的良好原料。")

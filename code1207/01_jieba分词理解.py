# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/7 10:23
Create User : 19410
Desc : jieba分词内部主要基于词典匹配实现，首先获取所有可能的分词方式，然后在所有分词方式选择最有可能的分词结果
"""
import os
import sys

print(sys.path)

# import jieba导包运行成功的前提是：jieba所在的文件夹必须在sys.path所对应的文件夹中
import jieba

print(jieba.__version__)

# 自定义词典
# jieba.load_userdict("./jieba.word")
jieba.load_userdict(os.path.join(os.path.dirname(__file__), "jieba.word"))

if __name__ == '__main__':
    # 文本分词
    print("  ".join(jieba.cut("我喜欢上学")))
    print(jieba.lcut("我喜欢上学"))
    print(jieba.lcut("送餐公司中饿了么是值得选择的"))

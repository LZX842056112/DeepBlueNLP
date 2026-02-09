# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/7 15:09
Create User : 19410
Desc : xxx
"""

def t0():
    from sklearn.feature_extraction.text import CountVectorizer
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    print(f"从训练数据中提取出来的单词列表:\n{vectorizer.get_feature_names_out()}\n")
    print(f"词袋法转换后的结果:\n{X.toarray()}\n")

def t1():
    from sklearn.feature_extraction.text import TfidfVectorizer
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    print(f"从训练数据中提取出来的单词列表:\n{vectorizer.get_feature_names_out()}\n")
    print(f"TF-IDF转换后的结果:\n{X.toarray().round(2)}\n")

if __name__ == '__main__':
    # t0()
    t1()


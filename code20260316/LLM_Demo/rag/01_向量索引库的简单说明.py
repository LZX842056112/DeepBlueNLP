# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/16 21:43
Create User : 19410
Desc : 向量索引库功能：提供基于向量相似度进行提取最相似的向量列表

pip install faiss-cpu==1.11.0
"""


import numpy as np
import faiss

dimension = 3
x = np.random.randn(100000, dimension)
x = x.astype(np.float32)

# 创建索引库
index = faiss.index_factory(
    dimension,  # 维度大小
    "HNSW8",  # 向量库的类型
    faiss.METRIC_L2  # 给定距离的表达形式/相似度的表达形式
)
index.train(x)
index.add(x)

# 查询搜索
k = 5
query_vectors = x[2:3]
D, I = index.search(query_vectors, k)  # D是距离矩阵，I是索引矩阵
print(D)
print(I)



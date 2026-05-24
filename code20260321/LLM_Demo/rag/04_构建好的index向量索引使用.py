# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/18 21:10
Create User : 19410
Desc : xxx
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

emb_model_path = r"D:\huggingface\modelscope\hub\thomas\m3e-base"

embedding_model = HuggingFaceEmbeddings(
    model_name=emb_model_path,
    encode_kwargs={
        "normalize_embeddings": True,  # 是否针对向量进行L2 Norm处理
        "batch_size": 4
    },
    show_progress=True
)
index = FAISS.load_local(
    'output/faiss_index/01', embedding_model,
    allow_dangerous_deserialization=True,
    normalize_L2=True,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
)

query = "孙悟空的师傅"
docs = index.similarity_search(query=query, k=10)
print(docs)

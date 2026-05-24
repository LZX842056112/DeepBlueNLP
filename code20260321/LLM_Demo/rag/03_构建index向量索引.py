# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/18 20:15
Create User : 19410
Desc : 针对原始的文档进行分片处理，然后调用向量模型获取每个片段的文本向量，最后基于文本向量构建索引
"""
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
# pip install langchain-huggingface==1.2.1
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy


def build_index():
    # 1. 原始数据加载
    print("原始数据加载....")
    in_folder = "./output/原文"
    docs = []
    for name in os.listdir(in_folder):
        if name.endswith(".txt"):
            file = os.path.join(in_folder, name)
            loader = TextLoader(file, encoding='utf-8')
            doc = loader.load()
            docs.extend(doc)
    print(f"总加载原始文档数目为:{len(docs)}  \n文档1: {len(docs[0].page_content)} - {docs[0]}\n")

    # 2. 文档分片处理
    print("开始文档分片处理...")
    emb_model_path = r"D:\huggingface\modelscope\hub\thomas\m3e-base"
    tokenizer = AutoTokenizer.from_pretrained(emb_model_path)
    spliter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=500,
        chunk_overlap=200,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    split_docs = []
    for doc in docs:
        new_docs = spliter.split_documents([doc])
        for new_doc in new_docs:
            split_docs.append(new_doc)
    print(f"分割后的片段数量:{len(split_docs)}  \n文档1: {len(split_docs[0].page_content)} - {split_docs[0]}")
    print(f"文档2: {len(split_docs[1].page_content)} - {split_docs[1]}\n")

    # 开始构建索引
    print("开始构造索引....")
    embedding_model = HuggingFaceEmbeddings(
        model_name=emb_model_path,
        encode_kwargs={
            "normalize_embeddings": True,  # 是否针对向量进行L2 Norm处理
            "batch_size": 4
        },
        show_progress=True
    )
    index = FAISS.from_documents(
        split_docs,
        embedding_model,
        normalize_L2=True,
        distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
    )
    print("索引构造完成")

    # 保存索引
    index.save_local('output/faiss_index/01')


if __name__ == '__main__':
    build_index()

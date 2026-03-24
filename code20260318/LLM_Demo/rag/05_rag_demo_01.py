# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/18 21:25
Create User : 19410
Desc : 单个操作RAG
"""
import warnings

warnings.filterwarnings('ignore')

from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate


class ChatClient(object):
    def __init__(self):
        emb_model_path = r"D:\cache\huggingface\hub\thomas\m3e-base"
        faiss_index_path = 'output/faiss_index/01'

        # 1. 恢复
        embedding_model = HuggingFaceEmbeddings(
            model_name=emb_model_path,
            encode_kwargs={
                "normalize_embeddings": True,  # 是否针对向量进行L2 Norm处理
                "batch_size": 4
            },
            show_progress=True
        )
        self.index: FAISS = FAISS.load_local(
            faiss_index_path, embedding_model,
            allow_dangerous_deserialization=True,
            normalize_L2=True,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
        )

        # 2. 获取OpenAI的接口
        # OpenAI API转发 cloudflare
        base_url = "https://gateway.ai.cloudflare.com/v1/67b8ebfcb6b836e009e1fb540f160fa5/nlp_0314/openrouter/v1"
        api_key = "sk-c3043bdd170c42fda7c98071d04b5cf9"
        model_name = "qwen/qwen3-235b-a22b-2507"
        max_tokens = None
        self.model: ChatOpenAI = ChatOpenAI(
            streaming=False,
            verbose=True,
            callbacks=None,
            openai_api_key=api_key,
            openai_api_base=base_url,
            model_name=model_name,
            temperature=0.9,
            max_tokens=max_tokens
        )

    def chat01(self):
        while True:
            msg = input("我:").strip()  # 孙悟空的师傅是谁？
            if msg == 'q':
                break
            if len(msg) == 0:
                continue
            _messages = [{'role': 'user', 'content': msg}]

            # 调用模型
            output = self.model.invoke(_messages)
            if isinstance(output, AIMessage):
                _ai_msg = output.content
            else:
                _ai_msg = output
            print(f"你:{_ai_msg}")

    def chat02(self):
        def _search_docs(_query):
            # 从index这个索引库中找出和query最相关的k个文本
            _docs = self.index.similarity_search(_query, k=5)
            _content = ""
            for _i, _doc in enumerate(_docs):
                _content += f"\n\n外部知识{_i}:{_doc.page_content}"
            return _content

        chat_message = ChatMessagePromptTemplate.from_template(
            """
            根据下面的问题和知识文档，给出一个全面的答案。
            只回答所问的问题，回答应该简洁且与问题相关。
            如果你无法从给定的知识文档中找到信息，那么直接"未找到相关信息"。
            
            问题:
            {{ query }}
            
            知识文档: 
            {{ context }}
                        """,
            "jinja2",
            role="user",
        )
        chat_prompt = ChatPromptTemplate.from_messages([chat_message])
        chain = chat_prompt | self.model
        # chain = LLMChain(prompt=chat_prompt, llm=self.model, memory=None)

        while True:
            query = input("我:").strip()  # 孙悟空的师傅是谁？
            if query == 'q':
                break
            if len(query) == 0:
                continue
            context = _search_docs(query)

            # 调用模型
            output = chain.invoke({
                "query": query,
                "context": context
            })
            if isinstance(output, AIMessage):
                _ai_msg = output.content
            else:
                _ai_msg = output
            print(f"你:{_ai_msg}")

    def chat03(self):
        def _search_docs(_query):
            # 从index这个索引库中找出和query最相关的k个文本
            from utils import down_web_page_content_with_query
            _docs = down_web_page_content_with_query(query, 5)
            _content = ""
            for _i, _doc in enumerate(_docs):
                _content += f"\n\n外部知识{_i}:{_doc.page_content}"
            return _content

        chat_message = ChatMessagePromptTemplate.from_template(
            """
根据下面的问题和知识文档，给出一个全面的答案。
只回答所问的问题，回答应该简洁且与问题相关。
如果你无法从给定的知识文档中找到信息，那么直接"未找到相关信息"。

问题:
{{ query }}

知识文档: 
{{ context }}
            """,
            "jinja2",
            role="user",
        )
        chat_prompt = ChatPromptTemplate.from_messages([chat_message])
        chain = chat_prompt | self.model
        # chain = LLMChain(prompt=chat_prompt, llm=self.model, memory=None)

        while True:
            query = input("我:").strip()  # 孙悟空的师傅是谁？
            if query == 'q':
                break
            if len(query) == 0:
                continue
            context = _search_docs(query)

            # 调用模型
            output = chain.invoke({
                "query": query,
                "context": context
            })
            if isinstance(output, AIMessage):
                _ai_msg = output.content
            else:
                _ai_msg = output
            print(f"你:{_ai_msg}")


if __name__ == '__main__':
    client = ChatClient()
    client.chat01()
    # client.chat02()
    # client.chat03()

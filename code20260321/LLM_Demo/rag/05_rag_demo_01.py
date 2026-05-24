# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/18 21:25
Create User : 19410
Desc : 单个操作RAG
"""
import json
import warnings

warnings.filterwarnings('ignore')

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate


class ChatClient(object):
    def __init__(self):
        emb_model_path = r"D:\cache\huggingface\modelscope\hub\thomas\m3e-base"
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
        api_key = "sk-or-v1-"
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
            # PS: 计算的是query和文档之间的相似度
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

知识文档: 
{{ context }}

问题:
{{ query }}
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

    def chat04(self):
        def _search_docs(_query):
            # 从index这个索引库中找出和query最相关的k个文本
            # 最原始的检索： 直接基于query检索
            _find_docs = self.index.similarity_search_with_score(_query, k=5)

            # query改写
            querys = []
            try:
                _query_promot = f"""
                    将下列问题的问法进行改写，使用中文，采用不同的语法结构进行询问，比如陈述句的形式：
                    返回结果仅包含改写后的问题，使用json字符串返回：
                    返回json格式为：
                    [
                    "问题1", "问题2", ....
                    ]

                    Question:
                    {_query}
                        """
                output = self.model.invoke(_query_promot, n=10)
                content = output.content.strip()
                querys = json.loads(content)
                print(f"改写的query列表:{querys}")
            except:
                pass

            def _filter_fn(_meta):
                for _doc, _ in _find_docs:
                    # 如果当前检索的知识的来源 和 docs(已有的知识)来源是一致的，那么当前检索到的知识丢弃
                    if _meta['source'] == _doc.metadata['source']:
                        return False
                return True

            for _query in querys:
                extend_docs = self.index.similarity_search_with_score(
                    _query, k=2,
                    filter=_filter_fn,
                    fetch_k=5 * 3
                )
                _find_docs.extend(extend_docs)

            # 一般情况下建议这里地方增加一个rerank的重排模型 -- 判断query和doc之间是否有关系

            _find_docs = sorted(_find_docs, key=lambda t: t[1])[:6]
            _content = ""
            for _i, (_doc, _score) in enumerate(_find_docs):
                print(f"当前文档相似度:{_i} {_score}")
                _content += f"\n\n外部知识{_i}:{_doc.page_content}"
            return _content

        chat_message = ChatMessagePromptTemplate.from_template(
            """
根据下面的问题和知识文档，给出一个全面的答案。
只回答所问的问题，回答应该简洁且与问题相关。
如果你无法从给定的知识文档中找到信息，那么直接"未找到相关信息"。

知识文档: 
{{ context }}

问题:
{{ query }}
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
    # client.chat01() # 直接用query调用LLM获取结果
    # client.chat02()  # 直接 index 向量相似度获取相似文本后再喂LLM获取结果
    # client.chat03() # 通过 search API 获取相关文本后，再喂LLM获取结果
    client.chat04()  # 改写query后进行向量检索

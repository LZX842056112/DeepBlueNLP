# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/21 15:34
Create User : 19410
Desc : 从互联网获取信息
"""

import logging

import requests
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field


def compress_content(content: str, query: str):
    try:
        from .utils import model_container

        llm = model_container.MODEL
        if llm is None:
            return content

        # 压缩一下content的内容
        messages = [
            {'role': 'system',
             'content': "你是一位专业的信息提取专家，从给定的content文本中提取出和query问题相关的原始文本描述，如果给定的content中无法回答query问题，那么允许直接返回空字符串。要求最终返回结果以简体中文进行返回。"},
            {'role': 'user', 'content': f"content: {content}\n\n#### query: {query}"}
        ]
        logging.info("开始进行互联网信息压缩....")
        output = llm.invoke(messages)
        if isinstance(output, AIMessage):
            content = output.content
        else:
            content = output
    except Exception as e:
        logging.error("进行互联网的信息压缩操作异常", exc_info=e)
    return content


def search_engine_iter(query: str):
    """

    :param query:
    :return:
    """
    header = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer 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'
    }
    # 检索urls
    data = {
        "query": query,
        "max_results": 3,
        "time_range": "week",
        "include_raw_content": "text"
    }
    response = requests.post(
        url='https://api.tavily.com/search',
        json=data,
        headers=header
    )
    content = ''
    if response.status_code == 200:
        datas = response.json()['results']
        raw_content = [
            f"资料{i}:{compress_content(data['raw_content'], query)}" for i, data in enumerate(datas) if
            data['raw_content'] is not None
        ]
        content = "\n\n".join(raw_content)
        # urls = [data['url'] for data in datas]
        #
        # data = {
        #     "urls": urls
        # }
        # response = requests.post(
        #     url='https://api.tavily.com/extract',
        #     json=data,
        #     headers=header
        # )
        # if response.status_code == 200:
        #     datas = response.json()['results']
        #     raw_content = [f"资料{i}:{data['raw_content']}" for i, data in enumerate(datas)]
        #     content = "\n\n".join(raw_content)
    return content


def search_internet(query: str):
    return search_engine_iter(query)


class SearchInternetInput(BaseModel):
    query: str = Field(description="在互联网上检索的query关键字")

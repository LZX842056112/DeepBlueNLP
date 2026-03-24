# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/9/13 16:04
Create User : 19410
Desc : xxx
"""
import logging
import re
from typing import List

import requests
import json

from langchain_core.documents import Document


def search_urls_with_internet(query, k=10):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
        "gl": "cn",
        "hl": "zh-cn",
        "num": max(k, 10)
    })
    headers = {
        'X-API-KEY': '50dbed9141280afbb52fc337bb886f025a76fc2d',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    datas = response.json()['organic']
    urls = [data['link'] for data in datas]
    return list(set(urls))


def down_web_content_with_urls(urls: List[str]) -> str:
    url = "https://api.tavily.com/extract"

    payload = {
        "urls": urls
    }
    headers = {
        'Authorization': 'Bearer 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',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, json=payload)
    datas = response.json()['results']
    return [data['raw_content'] for data in datas]


def down_web_page_content_with_query(query: str, k: int = 10) -> List[Document]:
    docs = []
    try:
        urls = search_urls_with_internet(query, k * 2)
        web_contents = down_web_content_with_urls(urls)
        for content in web_contents:
            content = re.sub(r'[^\u4e00-\u9fa5\s]', ' ', content)
            content = re.sub(r'\W+', ' ', content)
            docs.append(Document(page_content=content))
    except Exception as e:
        logging.error(f"基于url从互联网下载数据异常:{e}")
    return docs[:k]


if __name__ == '__main__':
    print(down_web_page_content_with_query("孙悟空的武器是什么"))

# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/5 21:23
Create User : 19410
Desc : xxx
"""

import requests

base_url = "http://118.178.146.32:9003"
base_url = "http://47.97.84.93:5000"
base_url = "http://120.26.175.28:9999"
base_url = "http://118.31.246.133:9999"


def get_call(url, params):
    response = requests.get(url, json=params)
    if response.status_code == 200:
        print(f"服务器返回结果为: {response.json()}")
    else:
        print(f"客户端和服务器之间的连接存在问题: {response.status_code}")


def post_call(url, params, with_json=False):
    if with_json:
        response = requests.post(url, json=params)
    else:
        response = requests.post(url, data=params)
    if response.status_code == 200:
        print(f"服务器返回结果为: {response.json()}")
    else:
        print(f"客户端和服务器之间的连接存在问题: {response.status_code}")


def tt01():
    get_call(
        url=fr"{base_url}/predict",
        params={}
    )
    get_call(
        url=fr"{base_url}/predict",
        params={
            'text': '请给我放一个王宝强的电影的幕后花絮',
            'top_k': 3
        }
    )
    post_call(
        url=fr"{base_url}/predict",
        params={
            'text': '帮我打开一下空调',
            'top_k': 3
        },
        with_json=True
    )

    post_call(
        url=fr"{base_url}/predict",
        params={
            'text': '记得明天5点提醒我戴耳机',
            'top_k': 4
        },
        with_json=True
    )


if __name__ == '__main__':
    tt01()

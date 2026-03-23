# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/14 15:20
Create User : 19410
Desc : xxx
"""
import requests


def call_worker_address():
    """
    curl -X 'POST' \
      '' \
      -H 'accept: application/json' \
      -d ''
    :return:
    """
    response = requests.post(
        url="http://8.136.193.178:20001/get_worker_address",
        json={
            # "model": "xinghuo-api"
            "model": "Qwen1___5-0___5B-Chat"
        }
    )
    if response.status_code == 200:
        print(response.json())


if __name__ == '__main__':
    call_worker_address()

# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/23 20:57
Create User : 19410
Desc : xxx
"""

import requests

# 使用api-key进行对应的工作流的区分
url = "http://120.27.200.181/v1/workflows/run"
headers = {
    'Authorization': 'Bearer app-HotrPzQSnN0gHLvXLA5tGSnZ',
    'Content-Type': 'application/json',
}
datas = {
    "inputs": {
        # "input_text": "酒店早餐味道不太好吃，晚上也有点吵",
        "input_text": "昨晚睡的非常舒服，早上的早餐也很好吃",
        "Multisentiment": "True"
    },
    # "response_mode": "streaming",
    "response_mode": "blocking",
    "user": "abc-123"
}

response = requests.post(url, headers=headers, json=datas)
if response.status_code == 200:
    print(response.json()['data']['outputs'])
else:
    print(f"异常:{response.status_code}")

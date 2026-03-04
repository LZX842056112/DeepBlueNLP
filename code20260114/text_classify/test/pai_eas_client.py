# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/14 22:14
Create User : 19410
Desc : xxx
"""
import json

# pip install -U eas-prediction

from eas_prediction import PredictClient
from eas_prediction import StringRequest, StringResponse

if __name__ == '__main__':
    client = PredictClient(
        'http://1757826125271350.cn-shenzhen.pai-eas.aliyuncs.com',
        'text_classify_nlp0114'
    )
    client.set_token('NDFlNTIxNmQ0OTcwZmI5ZGQ0OWEwZmM4NGRkMzA0NTdmMzdiNzJjMA==')
    client.init()

    request = StringRequest('{"text":"从张家界怎么去慈利，帮我规划一下路线", "top_k":5}')
    request = StringRequest('{"text":"今天好热呀", "top_k":5}')
    request = StringRequest('{"text":"房间里面好热呀", "top_k":5}')
    resp: StringResponse = client.predict(request)
    # print(resp)
    result = json.loads(resp.response_data)
    print(type(result))
    print(result)

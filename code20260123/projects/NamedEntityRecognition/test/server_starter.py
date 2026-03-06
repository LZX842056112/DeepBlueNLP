# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/17 11:05
Create User : 19410
Desc : xxx
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

if __name__ == '__main__':
    from ner.deploy.fastapi_app_onnx import start_server

    start_server(
        # model_path="./dsw_output/medical/bert/models/best.onnx",
        # model_path="./output/china-people-daily-ner-corpus/bert/models/best.onnx",
        model_path="./output/travel_query/bert/models/best.onnx",
        host="0.0.0.0",
        port=9001
    )

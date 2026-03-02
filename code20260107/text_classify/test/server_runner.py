# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/5 21:44
Create User : 19410
Desc : xxx
"""

import os
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")))


def start_flask_jit():
    from text_classify.deploy.flask_app import start_server

    os.environ['MODEL_PATH'] = './output/intention/bert1/deploy/best.pt'

    start_server(
        # model_path='./output/intention/bert1/deploy/best.pt',
        # host="0.0.0.0",
        # port=5000
    )


def start_flask_onnx():
    from text_classify.deploy.flask_app_onnx import start_server

    os.environ['MODEL_PATH'] = './output/intention/bert1/deploy/best.onnx'

    start_server(
        # model_path='./output/intention/bert1/deploy/best.pt',
        # host="0.0.0.0",
        # port=5000
    )


def start_fastapi_onnx():
    from text_classify.deploy.fastapi_app_onnx import start_server

    os.environ['MODEL_PATH'] = './output/intention/bert1/deploy/best.onnx'

    start_server(
        # model_path='./output/intention/bert1/deploy/best.pt',
        # host="0.0.0.0",
        # port=5000
    )


if __name__ == '__main__':
    # start_flask_onnx()
    start_fastapi_onnx()

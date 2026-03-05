# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/7 21:42
Create User : 19410
Desc : xxx
"""

import logging
import os
from typing import Optional

import uvicorn
from fastapi import FastAPI, Body

from .onnx_predictor import Predictor

# 构建应用
app = FastAPI()
# 定义一个变量
predictor: Optional[Predictor] = None


@app.get("/")
def index():
    return "基于FastAPI的算法API部署后端框架 + 命名实体识别 + ONNX"


async def predict(
        text: str = Body(..., description="待预测的文本text"),
        empty_param: int = Body(1, description="占位用的参数")
):
    try:
        # 2. 参数的检查、过滤、转换
        if text is None or len(text) == 0:
            return {'code': 2, 'msg': f'请求参数异常，请给定有效请求参数 [{text}]'}

        # 3. 调用模型获取预测结果
        pred_result = predictor.predict(
            x=text
        )

        # 4. 将模型预测结果转换返回给调用方
        return {'code': 0, 'msg': '成功', 'data': pred_result, 'text': text}
    except Exception as e:
        error_msg = f"服务器后端异常 {e}"
        logging.error(error_msg, exc_info=e)
        return {'code': 1, 'msg': error_msg}


def start_server(model_path=None, host="0.0.0.0", port=9001):
    global predictor

    app.get("/predict", summary="命名实体识别")(predict)
    app.post("/predict", summary="命名实体识别")(predict)

    predictor = Predictor(
        onnx_model_path=model_path or os.environ['MODEL_PATH']
    )
    # http://127.0.0.1:9001/docs
    uvicorn.run(app, host=host, port=port, log_level="info")

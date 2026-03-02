# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/7 21:28
Create User : 19410
Desc : FastAPI案例
"""
import logging

import uvicorn
from fastapi import FastAPI, Body

# 构建应用
app = FastAPI()


@app.get("/")
def index():
    return "基于FastAPI的算法API部署后端框架"


@app.get("/fetch_order/{order_id}")
def fetch_order(order_id: str):
    return {"order_id": order_id, "order_name": f"订单_{order_id}"}


async def predict(
        text: str = Body(..., description="待预测的文本text"),
        top_k: int = Body(1, description="获取预测概率最大的前K个值")
):
    try:
        # 2. 参数的检查、过滤、转换
        if text is None or len(text) == 0:
            return {'code': 2, 'msg': f'请求参数异常，请给定有效请求参数 [{text}]'}
        if top_k <= 0:
            top_k = 1

        # 3. 调用模型获取预测结果
        pred_result = {}

        # 4. 将模型预测结果转换返回给调用方
        return {'code': 0, 'msg': '成功', 'data': pred_result, 'text': text}
    except Exception as e:
        error_msg = f"服务器后端异常 {e}"
        logging.error(error_msg, exc_info=e)
        return {'code': 1, 'msg': error_msg}


app.get("/predict", summary="文本分类预测方法")(predict)
app.post("/predict", summary="文本分类预测方法")(predict)

if __name__ == '__main__':
    # http://127.0.0.1:9001/docs
    uvicorn.run(app, host="0.0.0.0", port=9001, log_level="info")

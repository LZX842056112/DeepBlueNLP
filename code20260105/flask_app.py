# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/5 21:08
Create User : 19410
Desc : 模型部署的Flask后端接口通用程序
"""
import logging

from flask import Flask, jsonify, request
# pip install flask-cors
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # 支持跨域访问


@app.route("/")
def index():
    return "基于Flask的算法API部署后端框架"


@app.route("/text_classify/intent/predict", methods=['GET', 'POST'])
def predict():
    try:
        # 1. 获取参数
        _args = None
        if request.method == 'GET':
            print(f"当前请求方式为: GET -- {request.args}")
            # 当请求方式为GET的时候，参数直接从request.args(args是一个字典)中按照参数名称直接获取即可
            _args = request.args
        elif request.method == 'POST':
            print(f"当前请求方式为：POST -- {request.form}")
            # 当请求方式为POST的时候，一般的参数直接从request.form(form是一个字典)中按照参数名称直接获取即可
            # 当请求方式为POST的时候，如果传递的数据为文件的话，需要从request.files中获取
            _args = request.form
            if len(_args) == 0:
                try:
                    _args = request.get_json()
                    print(f"POST JSON 参数 {_args}")
                except Exception as e:
                    raise ValueError(f"从request json中获取参数数据异常 {e}")
            pass
        else:
            # 这部分代码不会触发的
            raise ValueError("当前服务仅支持GET和POST请求方式!")
        if _args is None:
            raise ValueError("获取请求参数对象为None， 请检查!")
        text = _args.get('text')
        top_k = int(_args.get('top_k', '1'))

        # 2. 参数的检查、过滤、转换
        if text is None or len(text) == 0:
            return jsonify({'code': 2, 'msg': f'请求参数异常，请给定有效请求参数 [{text}]'})
        if top_k <= 0:
            top_k = 1

        # 3. 调用模型获取预测结果
        pred_result = {}

        # 4. 将模型预测结果转换返回给调用方
        return jsonify({'code': 0, 'msg': '成功', 'data': pred_result, 'text': text})
    except Exception as e:
        error_msg = f"服务器后端异常 {e}"
        logging.error(error_msg, exc_info=e)
        return jsonify({'code': 1, 'msg': error_msg})


if __name__ == '__main__':
    app.run(
        host="0.0.0.0",
        port=5001
    )

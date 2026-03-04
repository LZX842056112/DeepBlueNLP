# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/14 21:44
Create User : 19410
Desc : xxx
"""
import json
import logging
import os.path

# pip install http://eas-data.oss-cn-shanghai.aliyuncs.com/sdk/allspark-0.13-py2.py3-none-any.whl
import allspark


class MyProcessor(allspark.BaseProcessor):
    """ MyProcessor is an example
        you can send mesage like this to predict
        curl -v http://127.0.0.1:8080/api/predict/service_name -d '2 105'
    """

    def initialize(self):
        """ load module, executed once at the start of the service
             do service initialization and load models in this function.
            定义模型恢复的相关代码 仅调用一次
        """
        from text_classify.deploy.onnx_predictor import Predictor

        model_dir = os.path.abspath(os.environ['MODEL_DIR'])
        print(f"开始恢复模型参数:{model_dir}")
        # noinspection PyAttributeOutsideInit
        self.predictor = Predictor(onnx_model_path=os.path.join(model_dir, "best.onnx"))


    def process(self, data):
        """ process the request data
        每次请求，均会执行该方法
            : data 二进制字符串 --> 框架定义的
                PS: data是一个json字符串，{"text":"xxxx", "top_k":23} --> 业务定义
        """
        try:
            text = str(data, encoding='utf-8')
            record = json.loads(text)
            result = self.predictor.predict(x = record['text'], k=int(record.get('top_k', '1')))
            output = {'code': 0, 'msg': '成功', 'data': result, 'request': record}
        except Exception as e:
            logging.error(f"服务器异常:{e}", exc_info=e)
            output = {'code': 1, 'msg': f'服务器异常:{e}'}
        result = json.dumps(output, ensure_ascii=False)
        return bytes(result, encoding='utf8'), 200


if __name__ == '__main__':
    # allspark.default_properties().put('rpc.keepalive', '10000')
    # 设置服务计算超时时间为10s, 默认为5秒
    # parameter worker_threads indicates concurrency of processing
    endpoint = os.environ['PAI_ENDPOINT']
    runner = MyProcessor(worker_threads=10, endpoint=endpoint)
    runner.run()

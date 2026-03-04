FROM registry.cn-shenzhen.aliyuncs.com/gerry_ai/deploy:python_3.12

# 容器内安装环境
RUN pip install flask==3.1.1 flask-cors==6.0.2 transformers==4.57.3 jieba onnxruntime==1.22.0 fastapi==0.109.0 uvicorn==0.40.0 -i http://mirrors.cloud.aliyuncs.com/pypi/simple/ --trusted-host mirrors.cloud.aliyuncs.com

# 复制COPY文件到容器内部
COPY src /opt/text_classify/src
COPY test /opt/text_classify/test

# 对外暴露的端口号
EXPOSE 9001

# 启动命令(容器的执行命令)
CMD python /opt/text_classify/test/server_runner.py
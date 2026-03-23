# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/14 16:37
Create User : 19410
Desc : 调用OpenAI格式的模型接口
"""
import copy
from datetime import datetime


def invoke_with_http():
    """
    通过http请求的方式来调用接口
    :return:
    """
    import requests

    url = "http://121.196.166.48:20000"
    api_key = None
    model_name = "Qwen1___5-0___5B-Chat"
    model_name = "xinghuo-api"

    # url = 'https://api.deepseek.com'
    # api_key = 'sk-1c818722cc70442a9010695d4b89e7e0'  # 这个会删除的
    # model_name = "deepseek-chat"
    #
    # # 一个免费的大模型的网站: https://openrouter.ai/
    # url = 'https://openrouter.ai/api'
    # api_key = "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"
    # model_name = "qwen/qwen3-235b-a22b-2507"

    # # OpenAI API转发 cloudflare
    # url = "https://gateway.ai.cloudflare.com/v1/50a8b09f35bc8aa57f135237aced6285/nlp_20260320/deepseek"
    # api_key = 'sk-1c818722cc70442a9010695d4b89e7e0'  # 这个会删除的
    # model_name = "deepseek-chat"
    #
    # # OpenAI API转发 cloudflare
    # url = "https://gateway.ai.cloudflare.com/v1/50a8b09f35bc8aa57f135237aced6285/nlp_20260320/openrouter"
    # api_key = "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"
    # model_name = "qwen/qwen3-235b-a22b-2507"

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    if api_key is not None:
        headers['Authorization'] = f'Bearer {api_key}'

    def _list_models():
        try:
            _url = f"{url}/v1/models"
            _response = requests.get(_url, headers=headers)
            if _response.status_code == 200:
                print(f"list_models结果为: {_response.json()}")
            else:
                print(f"list_models异常:{_response.status_code}")
        except Exception as e:
            print(f"获取模型列表异常:{e}")

    def _chat_with_llm(_model_name=None):
        _model_name = _model_name or "Qwen1___5-0___5B-Chat"
        _url = f"{url}/v1/chat/completions"

        _messages = []
        while True:
            msg = input("我:")
            if msg == 'q':
                break
            _messages.append({'role': 'user', 'content': msg})
            if len(_messages) > 5:
                # 针对过去的历史信息仅保留部分 --> 保留最近3轮的
                _messages = _messages[-5:]

            _data = {
                "model": _model_name,
                "messages": _messages,
                "temperature": 0.7,
                "stream": False  # 非流式返回结果
            }
            _response = requests.post(
                _url, json=_data, headers=headers
            )
            if _response.status_code == 200:
                _result = _response.json()
                # print(f"完成返回的数据为:{_result}")
                _ai_role = _result['choices'][-1]['message']['role']
                _ai_msg = _result['choices'][-1]['message']['content']
                print(f"chat结果: {_ai_msg}")
                _messages.append({'role': _ai_role, 'content': _ai_msg})
            else:
                print(f"chat结果异常:{_response.status_code}")
                _messages = _messages[:-1]

    _list_models()
    _chat_with_llm(model_name)


def invoke_with_openai():
    #  pip install openai==1.106.0
    from openai import OpenAI

    # 我们自己部署的一个OpenAI规范的接口
    base_url = "http://121.196.166.48:20000/v1"
    api_key = "Empty"
    model_name = "Qwen1___5-0___5B-Chat"
    model_name = "xinghuo-api"

    base_url = 'https://openrouter.ai/api/v1'
    api_key = "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"
    model_name = "qwen/qwen3-235b-a22b-2507"

    # OpenAI API转发 cloudflare
    base_url = "https://gateway.ai.cloudflare.com/v1/50a8b09f35bc8aa57f135237aced6285/nlp_20260320/openrouter/v1"
    api_key = "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"
    model_name = "qwen/qwen3-235b-a22b-2507"

    # 创建一个客户端
    client = OpenAI(base_url=base_url, api_key=api_key)

    def _chat_with_llm(_model_name=None):
        _messages = []

        while True:
            msg = input("我:")
            if msg == 'q':
                break
            _messages.append({'role': 'user', 'content': msg})
            if len(_messages) > 5:
                # 将历史信息部分删除
                _messages = _messages[-5:]

            # 调用模型
            completion = client.chat.completions.create(
                extra_headers={
                    'k': 'v'
                },  # 给定额外的headers信息 -- http请求时候的header信息
                extra_body={
                    'bk': 'bv'
                },  # 额外的请求参数
                model=_model_name,
                messages=_messages
            )
            _ai_msg = completion.choices[0].message.content
            print(f"你:{_ai_msg}")
            _messages.append({'role': 'assistant', 'content': _ai_msg})

    _chat_with_llm(model_name)


def invoke_with_langchain():
    # pip install langchain_core==0.3.76 langchain_community==0.3.15
    from langchain_community.chat_models import ChatOpenAI
    from langchain_core.messages import AIMessage

    # 我们自己部署的一个OpenAI规范的接口
    base_url = "http://121.196.166.48:20000/v1"
    api_key = "Empty"
    model_name = "Qwen1___5-0___5B-Chat"
    # model_name = "xinghuo-api"
    max_tokens = 256

    # base_url = 'https://openrouter.ai/api/v1'
    # api_key = "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"
    # model_name = "qwen/qwen3-235b-a22b-2507"
    # max_tokens = None

    # OpenAI API转发 cloudflare
    # base_url = "https://gateway.ai.cloudflare.com/v1/50a8b09f35bc8aa57f135237aced6285/nlp_20260320/openrouter/v1"
    # api_key = "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"
    # model_name = "qwen/qwen3-235b-a22b-2507"
    # max_tokens = None

    # # OpenAI API转发 cloudflare
    # base_url = "https://gateway.ai.cloudflare.com/v1/50a8b09f35bc8aa57f135237aced6285/nlp_20260320/deepseek/v1"
    # api_key = 'sk-1c818722cc70442a9010695d4b89e7e0'  # 这个会删除的
    # model_name = "deepseek-chat"
    # max_tokens = None

    model = ChatOpenAI(
        streaming=False,
        verbose=True,
        callbacks=None,
        openai_api_key=api_key,
        openai_api_base=base_url,
        model_name=model_name,
        temperature=0.9,
        max_tokens=max_tokens
    )

    # 定义system message(系统固定属性)
    system = "你是一个非常棒的智能助手!"

    def _chat_with_llm():
        _messages = []

        while True:
            msg = input("我:")
            if msg == 'q':
                break
            _messages.append({'role': 'user', 'content': msg})
            if len(_messages) > 5:
                # 将历史信息部分删除
                _messages = _messages[-5:]

            # 构造最终调用模型的message
            _input_msgs = copy.deepcopy(_messages)
            # 添加固定的信息 -- 一般添加到最前面
            _input_msgs.insert(
                0,
                {'role': 'system',
                 'content': f"{system}\n- 当前时间为: {datetime.now().strftime('%Y-%m-%d %H:%M:%S %A')}\n- 回答的语气:俏皮一点"
                 }
            )

            # 调用模型
            output = model.invoke(_input_msgs)
            if isinstance(output, AIMessage):
                _ai_msg = output.content
            else:
                _ai_msg = output
            print(f"你:{_ai_msg}")
            _messages.append({'role': 'assistant', 'content': _ai_msg})

    _chat_with_llm()


def invoke_with_langchain_intention():
    # pip install langchain_core==0.3.76 langchain_community==0.3.15
    from langchain_community.chat_models import ChatOpenAI
    from langchain_core.messages import AIMessage

    # 我们自己部署的一个OpenAI规范的接口
    base_url = "http://121.196.166.48:20000/v1"
    api_key = "Empty"
    model_name = "Qwen1___5-0___5B-Chat"
    # model_name = "xinghuo-api"
    max_tokens = 256

    # base_url = 'https://openrouter.ai/api/v1'
    # api_key = "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"
    # model_name = "qwen/qwen3-235b-a22b-2507"
    # max_tokens = None

    # # OpenAI API转发 cloudflare
    # base_url = "https://gateway.ai.cloudflare.com/v1/50a8b09f35bc8aa57f135237aced6285/nlp_20260320/openrouter/v1"
    # api_key = "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"
    # model_name = "qwen/qwen3-235b-a22b-2507"
    # max_tokens = None

    # # OpenAI API转发 cloudflare
    # base_url = "https://gateway.ai.cloudflare.com/v1/50a8b09f35bc8aa57f135237aced6285/nlp_20260320/deepseek/v1"
    # api_key = 'sk-1c818722cc70442a9010695d4b89e7e0'  # 这个会删除的
    # model_name = "deepseek-chat"
    # max_tokens = None

    model = ChatOpenAI(
        streaming=False,
        verbose=True,
        callbacks=None,
        openai_api_key=api_key,
        openai_api_base=base_url,
        model_name=model_name,
        temperature=0.9,
        max_tokens=max_tokens
    )

    # 定义system message(系统固定属性)
    system = """
针对能够准确判断意图的文本，请直接从名称列表中获取最有可能的所属意图名称字符串，不要做任何更改，直接返回。如果没办法判断意图，直接返回Other。\n
意图名称列表:[Travel-Query, Music-Play, FilmTele-Play, Video-Play, Radio-Listen, HomeAppliance-Control, Weather-Query, Alarm-Update, Calendar-Query, TVProgram-Play, Audio-Play, Other]
    """
    examples = [
        {'role': 'user', 'content': '现在将空调开启制热模式吧'},
        {'role': 'assistant', 'content': 'HomeAppliance-Control'},
        {'role': 'user', 'content': '明天紫外线怎么样'},
        {'role': 'assistant', 'content': 'Weather-Query'},
    ]

    def _chat_with_llm():
        while True:
            msg = input("我:")
            if msg == 'q':
                break

            # 构造最终调用模型的message
            _input_msgs = copy.deepcopy(examples)
            _input_msgs.append({'role': 'user', 'content': msg})
            # 添加固定的信息 -- 一般添加到最前面
            _input_msgs.insert(0, {'role': 'system', 'content': system})

            # 调用模型
            output = model.invoke(_input_msgs)
            if isinstance(output, AIMessage):
                _ai_msg = output.content
            else:
                _ai_msg = output
            print(f"你:{_ai_msg}")

    _chat_with_llm()


if __name__ == '__main__':
    # invoke_with_http()
    # invoke_with_openai()
    # invoke_with_langchain()
    invoke_with_langchain_intention()

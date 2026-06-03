# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/21 17:38
Create User : 19410
Desc : 使用简化的代码
"""
import requests
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def add(a, b):
    """
    计算两个数的和
    :param a: 加数1
    :param b: 加数2
    :return:
    """
    print(f"计算加法:{a} + {b}")
    return float(a) + float(b)


@tool
def mul(a, b):
    """
    计算两个数的乘积
    :param a: 第一个乘数
    :param b: 第二个乘数
    :return:
    """
    print(f"计算乘法:{a} * {b}")
    return float(a) * float(b)


@tool
def weather(location: str):
    """
    获取对应城市的天气情况
    :param location:
    :return:
    """
    api_key = "SqWCDI5TuUyD4Nbby"
    url = f"https://api.seniverse.com/v3/weather/now.json?key={api_key}&location={location}&language=zh-Hans&unit=c"
    print(f"天气查询:{url}")
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        weather = {
            "temperature": data["results"][0]["now"]["temperature"],
            "description": data["results"][0]["now"]["text"],
        }
        return weather
    else:
        weather = {
            "temperature": 25.0,
            "description": "晴，默认返回!",
        }
        return weather
        # raise Exception(
        #    f"Failed to retrieve weather: {response.status_code}")


def get_model():
    base_url = "https://gateway.ai.cloudflare.com/v1/67b8ebfcb6b836e009e1fb540f160fa5/nlp_0314/openrouter/v1"
    api_key = "sk-or-v1-"
    model_name = "qwen/qwen3-235b-a22b-2507"

    # # OpenAI API转发 deepseek
    # base_url = "https://gateway.ai.cloudflare.com/v1/67b8ebfcb6b836e009e1fb540f160fa5/nlp_0314/deepseek/v1"
    # model_name = "deepseek-chat"
    # api_key = "sk-"  # 后期会删除

    max_tokens = None
    return ChatOpenAI(
        streaming=False,
        verbose=True,
        callbacks=None,
        openai_api_key=api_key,
        openai_api_base=base_url,
        model_name=model_name,
        temperature=0.9,
        max_tokens=max_tokens
    )


def run():
    model = get_model()
    agent = create_agent(
        model=model,
        tools=[add, mul, weather],
        system_prompt="你是一个智能助手，可以访问多种工具和服务。"
                      "\n"
                      "使用指南：\n"
                      "1. 根据用户需求选择合适的工具\n"
                      "2. 如果工具调用失败，告知用户并建议替代方案\n"
                      "3. 保持回答简洁准确\n"
                      "4. 涉及敏感操作 (如写文件) 前先确认\n"
    )

    while True:
        query = input("我:").strip()  # 孙悟空的师傅是谁？ 孙悟空的两个师傅分别是谁？
        if query == 'q':
            break
        if len(query) == 0:
            continue

        # 调用模型
        output = agent.invoke({
            "messages": [HumanMessage(content=query)]
        })
        print(type(output))
        if isinstance(output, AIMessage):
            _ai_msg = output.content
        elif isinstance(output, dict):
            _ai_msg = output['messages'][-1].content
        else:
            _ai_msg = output
        print(f"你:{_ai_msg}")


if __name__ == '__main__':
    run()

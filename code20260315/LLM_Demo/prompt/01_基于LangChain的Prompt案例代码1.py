# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/15 15:16
Create User : 19410
Desc : xxx
"""
import asyncio
import json

from langchain.chains.llm import LLMChain
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI

from server.chat.utils import History


def get_model():
    # 我们自己部署的一个OpenAI规范的接口
    base_url = "http://121.196.166.48:20000/v1"
    api_key = "Empty"
    model_name = "Qwen1___5-0___5B-Chat"
    # model_name = "xinghuo-api"
    max_tokens = 256

    # base_url = 'https://openrouter.ai/api/v1'
    # api_key = "sk-or-v1-"
    # model_name = "qwen/qwen3-235b-a22b-2507"
    # max_tokens = None

    # # OpenAI API转发 cloudflare
    # base_url = "https://gateway.ai.cloudflare.com/v1/67b8ebfcb6b836e009e1fb540f160fa5/nlp_0314/openrouter/v1"
    # api_key = "sk-or-v1-"
    # model_name = "qwen/qwen3-235b-a22b-2507"
    # max_tokens = None

    # # OpenAI API转发 cloudflare
    base_url = "https://gateway.ai.cloudflare.com/v1/50a8b09f35bc8aa57f135237aced6285/nlp_20260320/deepseek/v1"
    api_key = 'sk-'  # 这个会删除的
    model_name = "deepseek-chat"
    max_tokens = None

    model = ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=None,
        openai_api_key=api_key,
        openai_api_base=base_url,
        model_name=model_name,
        temperature=0.9,
        max_tokens=max_tokens
    )
    return model


def get_intention_chain(model):
    system = SystemMessage(content="""
    针对能够准确判断意图的文本，请直接从名称列表中获取最有可能的所属意图名称字符串，不要做任何更改，直接返回。如果没办法判断意图，直接返回Other。\n 
    意图名称列表:[Travel-Query, Music-Play, FilmTele-Play, Video-Play, Radio-Listen, HomeAppliance-Control, Weather-Query, Alarm-Update, Calendar-Query, TVProgram-Play, Audio-Play, Other]
        """)
    examples = [
        {'role': 'user', 'content': '现在将空调开启制热模式吧'},
        {'role': 'assistant', 'content': 'HomeAppliance-Control'},
        {'role': 'user', 'content': '明天紫外线怎么样'},
        {'role': 'assistant', 'content': 'Weather-Query'},
    ]
    example_msgs = [History.from_data(h).to_msg_template(is_raw=False) for h in examples]
    prompt_template = '''{{ input }}'''
    input_msg = History(role="user", content=prompt_template).to_msg_template(False)
    chat_prompt = ChatPromptTemplate.from_messages([system] + example_msgs + [input_msg])
    chain = LLMChain(prompt=chat_prompt, llm=model, memory=None)
    return chain


def get_travel_query_entity_chain(model):
    system = '从当前用户给定的三个`包含的文本中提取实体单词，不允许从其它位置进行提取。\n实体类别列表为:["出发时间", "出发地", "目的地", "交通方式"]\n并以json格式的结果返回，json中包含实体类型和实体片段两个字段，不允许返回其它内容。\n比如返回的格式为:[{"entity_type":"出发时间","entity_span":"明天"},...]\n'
    system = SystemMessage(content=system)
    examples = [
        {'role': 'user', 'content': '```今天还有去北京的飞机票吗```'},
        {
            'role': 'assistant',
            'content': '[{"entity_type":"出发时间","entity_span":"今天"},{"entity_type":"交通方式","entity_span":"飞机"},{"entity_type":"目的地","entity_span":"北京"}]'
        },
        {'role': 'user', 'content': '```明天天气怎么样```'},
        {
            'role': 'assistant',
            'content': '[]'
        },
    ]
    example_msgs = [History.from_data(h).to_msg_template(is_raw=False) for h in examples]
    prompt_template = '''```{{ input }}```'''
    input_msg = History(role="user", content=prompt_template).to_msg_template(False)
    chat_prompt = ChatPromptTemplate.from_messages([system] + example_msgs + [input_msg])
    chain = LLMChain(prompt=chat_prompt, llm=model, memory=None)
    return chain


def run():
    model = get_model()
    intention_chain = get_intention_chain(model)
    travel_query_entity_chain = get_travel_query_entity_chain(model)

    while True:
        query = input("我:")
        if query == 'q':
            break

        # 1. 先意图判断
        intention_text = intention_chain.invoke({"input": query})
        if 'text' not in intention_text:
            print(f"你: 意图判断失败 - {intention_text}")
            continue
        intention_text = intention_text['text']
        if intention_text == 'Travel-Query':
            # 走交通查询的业务逻辑
            travel_query_entity = travel_query_entity_chain.invoke({"input": query})
            travel_query_entity = json.loads(travel_query_entity['text'])
            print(f"你: {travel_query_entity} -- 后续工作应该是基于查询出来的实体信息从数据库/接口获取对应的数据，再将数据输入到LLM中进行整合，最终输出最终的文本")
        elif intention_text == 'Music-Play':
            # 走音乐播放的相关业务逻辑
            print(f"你: 音乐播放 - {intention_text} -- 先获取音乐播放需要的相关实体，然后通过QQ音乐平台、网易云音乐平台等找到对应的歌曲，然后播放即可")
        else:
            print(f"你：意图为 - {intention_text}")

    # query = "上海到北京怎么走"
    # result = intention_chain({"input": query})  # 同步调用...
    # print(result['text'])


if __name__ == '__main__':
    run()

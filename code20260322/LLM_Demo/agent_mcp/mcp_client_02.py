# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/22 10:18
Create User : 19410
Desc : 使用MCP服务的客户端
"""
from datetime import timedelta

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI


def get_model():
    base_url = "https://gateway.ai.cloudflare.com/v1/67b8ebfcb6b836e009e1fb540f160fa5/nlp_0314/openrouter/v1"
    api_key = "sk-or-v1-"
    model_name = "qwen/qwen3-235b-a22b-2507"

    # # OpenAI API转发 deepseek
    # base_url = "https://gateway.ai.cloudflare.com/v1/67b8ebfcb6b836e009e1fb540f160fa5/nlp_0314/deepseek/v1"
    # model_name = "deepseek-chat"
    # api_key = "sk-202bdf9647f340e99d47edbcf6b97f88"  # 后期会删除

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


async def run():
    model = get_model()

    client = MultiServerMCPClient({
        'math': {
            "transport": "stdio",
            "command": "python",
            "args": [r".\servers\math_server.py"]
        },
        'weather': {
            "transport": "stdio",
            "command": "python",
            "args": [r".\servers\weather_server.py"]
        },
        'product': {
            "transport": "streamable-http",
            "url": "http://127.0.0.1:8000/mcp"
        },
        "pozansky-stock-server": {
            "transport": "streamable-http",
            "url": "https://mcp.api-inference.modelscope.net/3252b4929ec941/mcp",
            "timeout": timedelta(seconds=5),
        },
        "bing-cn-mcp-server": {
            "transport": "streamable-http",
            "url": "https://mcp.api-inference.modelscope.net/195bf9cf49cc4e/mcp",
            "timeout": timedelta(seconds=5),
        }
    })
    # 加载工具
    tools = await client.get_tools()

    # 创建agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="你是一个智能助手，可以访问多种工具和服务。"
                      "\n"
                      "使用指南：\n"
                      "1. 根据用户需求选择合适的工具。如果有提供的工具，必须选择使用工具，不允许自己计算。\n"
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
        output = await agent.ainvoke({
            "messages": query
        })
        _ai_msg = output['messages'][-1].content
        print(f"你:{_ai_msg}")


if __name__ == '__main__':
    import asyncio

    asyncio.run(run())

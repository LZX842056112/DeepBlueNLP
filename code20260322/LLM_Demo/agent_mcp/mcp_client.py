# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/22 10:18
Create User : 19410
Desc : 使用MCP服务的客户端
"""
from langchain.agents import create_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from mcp import StdioServerParameters, stdio_client, ClientSession


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


async def run():
    model = get_model()

    # 启动参数
    server_params = StdioServerParameters(
        command="python",
        args=[r"D:\Projects\DeepBule\DeepBlueNLP\code20260322\LLM_Demo\agent_mcp\servers\math_server.py"]
    )

    async with stdio_client(server_params) as (read, writer):
        async with ClientSession(read, writer) as sess:
            # 初始化链接
            await sess.initialize()

            # 加载工具
            tools = await load_mcp_tools(sess)

            # 创建agent
            agent = create_agent(
                model=model,
                tools=tools,
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
                output = await agent.ainvoke({
                    "messages": query
                })
                _ai_msg = output['messages'][-1].content
                print(f"你:{_ai_msg}")


if __name__ == '__main__':
    import asyncio

    asyncio.run(run())

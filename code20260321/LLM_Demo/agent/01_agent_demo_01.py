# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/21 14:31
Create User : 19410
Desc : 参考Langchain-Chatchat内部的代码逻辑整理出来的agent应用代码
"""
from langchain_classic.agents import LLMSingleActionAgent, AgentExecutor
from langchain_classic.chains.llm import LLMChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from custom_template import CustomPromptTemplate, CustomOutputParser
from tools.utils import model_container
from tools_select import tools, tool_names

# 定义一个提示词模版字符串
AGENT_PROMPTS = {
    "default":
        'Answer the following questions as best you can. If it is in order, you can use some tools appropriately. '
        'You have access to the following tools:\n\n'
        '{tools}\n\n'
        'Use the following format:\n'
        'Question: the input question you must answer1\n'
        'Thought: you should always think about what to do and what tools to use.\n'
        'Action: the action to take, should be one of [{tool_names}]\n'
        'Action Input: the input to the action\n'
        'Observation: the result of the action\n'
        '... (this Thought/Action/Action Input/Observation can be repeated zero or more times)\n'
        'Thought: I now know the final answer\n'
        'Final Answer: the final answer to the original input question \n'
        'Begin!\n\n'
        'history: {history}\n\n'
        'Question: {input}\n\n'
        'Thought: {agent_scratchpad}'
}


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
    model_container.MODEL = model  # agent内部使用的llm模型对象
    prompt_template = AGENT_PROMPTS.get('default')

    # 构造一个输入的模版对象：会将所有的相关信息按照该对象中定义的方法进行转换为最终调用模型的字符串
    prompt_template_agent = CustomPromptTemplate(
        template=prompt_template,
        tools=tools,
        input_variables=["input", "intermediate_steps", "history"]
    )

    # 构建执行链路对象 --> 内部会自动调用对应方法将prompt转换为字符串
    llm_chain = LLMChain(llm=model, prompt=prompt_template_agent)
    # 构造输出数据的解析器
    output_parser = CustomOutputParser()
    # 构造一个agent对象
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:", "Observation"]
    )
    memory = ConversationBufferWindowMemory(k=2)  # 用来存储历史信息的
    # 构造一个agent的执行器
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
    )

    while True:
        msg = input("我:")
        if msg == 'q':
            break

        output = agent_executor(msg)
        if isinstance(output, AIMessage):
            print(f"你:{output.content}")
        elif isinstance(output, dict):
            print(f"你:{output['output']}")
        else:
            print(f"你:{output}")


if __name__ == '__main__':
    run()

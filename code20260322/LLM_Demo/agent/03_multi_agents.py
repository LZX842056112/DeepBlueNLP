# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/22 11:41
Create User : 19410
Desc : 基于Langchain-graph库的多智能体编排
"""
from langchain_openai import ChatOpenAI
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from typing import TypedDict


# 定义状态
class State(TypedDict):
    user_input: str
    answer: str
    query_type: str


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


llm = get_model()


# ######## 定义节点 #########
# 定义选择节点
def select_node(state: State) -> dict:
    prompt = f"""
    你是一个AI智能体，你的任务是判断用户输入的是什么学科类型的信息，有物理，数学，历史这三种类型。
    如果是物理类型的问题就回答"物理"，如果是数学类型的问题就回答"数学"，如果是历史相关就回答"历史"
    如果都不是则返回"其他"
    用户输入如下：{state['user_input']}
    """
    res = llm.invoke(prompt)

    if "物理" in res.content:
        return {"query_type": "physics"}
    elif "数学" in res.content:
        return {"query_type": "math"}
    elif "历史" in res.content:
        return {"query_type": "history"}
    else:
        return {"query_type": "other"}


# 定义物理节点
class PhysicsAgent(object):
    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def run(self, state: State):
        physics_template = f"""
            你是一位物理学家，擅长回答物理相关的问题，当你不知道问题的答案时，你就回答不知道。
            具体问题如下：
            {state['user_input']}
            """
        res = self.llm.invoke(physics_template)
        return {"answer": res.content}


# 数学节点
def math(state: State) -> dict:
    math_template = f"""
    你是一个数学家，擅长回答数学相关的问题，当你不知道问题的答案时，你就回答不知道。
    具体问题如下：
    {state['user_input']}
    """
    res = llm.invoke(math_template)
    return {"answer": res.content}


# 历史节点
def history(state: State) -> dict:
    history_template = f"""
    你是一个非常厉害的历史老师，擅长回答历史相关的问题，当你不知道问题的答案时，你就回答不知道。
    具体问题如下：
    {state['user_input']}
    """
    res = llm.invoke(history_template)
    return {"answer": res.content}


def other(state: State) -> dict:
    return {"answer": "其他类型，不做处理"}


def router(state: State):
    if state["query_type"] == "physics":
        return "physics"
    elif state["query_type"] == "math":
        return "math"
    elif state["query_type"] == "history":
        return "history"
    else:
        return "other"


if __name__ == '__main__':
    graph = StateGraph(State)
    physics_agent = PhysicsAgent(llm)

    # 添加节点
    graph.add_node("select_node", select_node)
    graph.add_node("physics", physics_agent.run)
    graph.add_node("math", math)
    graph.add_node("history", history)
    graph.add_node("other", other)

    # 添加边
    graph.add_edge(START, "select_node")
    graph.add_conditional_edges("select_node", router, {
        "physics": "physics",
        "math": "math",
        "history": "history",
        "other": "other"
    })

    graph.add_edge("physics", END)
    graph.add_edge("math", END)
    graph.add_edge("history", END)
    graph.add_edge("other", END)

    app = graph.compile()
    pic = app.get_graph().draw_mermaid_png()
    with open('./graph_pic.png', 'wb') as f:
        f.write(pic)

    while True:
        msg = input("我:").strip()
        if msg == 'q':
            break
        res = app.stream({"user_input": msg})
        for i in res:
            print(i)
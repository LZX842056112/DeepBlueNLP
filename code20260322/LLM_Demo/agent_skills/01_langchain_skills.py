# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/22 14:21
Create User : 19410
Desc : 在langchain框架中类似实现skills的这种渐进式加载逻辑
"""
import uuid
from typing import TypedDict, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver


class Skill(TypedDict):
    """可向智能体渐进式披露的技能。"""
    name: str  # 技能的唯一标识符
    description: str  # 1–2 句描述，用于系统提示词
    content: str  # 包含详细指令的完整技能内容


SKILLS: list[Skill] = [
    {
        "name": "sales_analytics",
        "description": "用于销售数据分析的数据库结构和业务逻辑，包括客户、订单和收入。",
        "content": """# 销售分析数据结构

## 表结构

### customers（客户表）
- customer_id（主键）
- name（姓名）
- email（邮箱）
- signup_date（注册日期）
- status（状态：active/active 表示活跃，inactive 表示非活跃）
- customer_tier（客户等级：bronze/银牌/silver/金牌/gold/白金/platinum）

### orders（订单表）
- order_id（主键）
- customer_id（外键 → customers）
- order_date（下单日期）
- status（订单状态：pending/待处理、completed/已完成、cancelled/已取消、refunded/已退款）
- total_amount（订单总金额）
- sales_region（销售区域：north/北区、south/南区、east/东区、west/西区）

### order_items（订单明细表）
- item_id（主键）
- order_id（外键 → orders）
- product_id（产品ID）
- quantity（数量）
- unit_price（单价）
- discount_percent（折扣百分比）

## 业务逻辑

**活跃客户**：status = 'active' 且 signup_date <= 当前日期 - 90天

**收入计算**：仅统计 status = 'completed' 的订单。orders 表中的 total_amount 字段已包含折扣后的金额。

**客户生命周期价值（CLV）**：某客户所有已完成订单金额的总和。

**高价值订单**：total_amount > 1000 的订单

## 示例查询

-- 获取最近一个季度收入排名前10的客户
SELECT
    c.customer_id,
    c.name,
    c.customer_tier,
    SUM(o.total_amount) as total_revenue
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
  AND o.order_date >= CURRENT_DATE - INTERVAL '3 months'
GROUP BY c.customer_id, c.name, c.customer_tier
ORDER BY total_revenue DESC
LIMIT 10;
""",
    },
    {
        "name": "inventory_management",
        "description": "用于库存跟踪的数据库结构和业务逻辑，包括产品、仓库和库存水平。",
        "content": """# 库存管理数据结构

## 表结构

### products（产品表）
- product_id（主键）
- product_name（产品名称）
- sku（库存单位编码）
- category（类别）
- unit_cost（单位成本）
- reorder_point（再订货点：库存低于此值时需补货）
- discontinued（是否停产：布尔值）

### warehouses（仓库表）
- warehouse_id（主键）
- warehouse_name（仓库名称）
- location（位置）
- capacity（容量）

### inventory（库存表）
- inventory_id（主键）
- product_id（外键 → products）
- warehouse_id（外键 → warehouses）
- quantity_on_hand（当前库存数量）
- last_updated（最后更新时间）

### stock_movements（库存变动记录表）
- movement_id（主键）
- product_id（外键 → products）
- warehouse_id（外键 → warehouses）
- movement_type（变动类型：inbound/入库、outbound/出库、transfer/调拨、adjustment/调整）
- quantity（数量：入库为正，出库为负）
- movement_date（变动日期）
- reference_number（参考编号）

## 业务逻辑

**可用库存**：inventory 表中 quantity_on_hand > 0 的记录

**需要补货的产品**：所有仓库中该产品的总库存量 ≤ 该产品的 reorder_point

**仅考虑在售产品**：除非特别分析停产商品，否则应排除 discontinued = true 的产品

**库存估值**：每个产品的 quantity_on_hand × unit_cost

## 示例查询

-- 查找所有仓库中库存低于再订货点的产品
SELECT
    p.product_id,
    p.product_name,
    p.reorder_point,
    SUM(i.quantity_on_hand) as total_stock,
    p.unit_cost,
    (p.reorder_point - SUM(i.quantity_on_hand)) as units_to_reorder
FROM products p
JOIN inventory i ON p.product_id = i.product_id
WHERE p.discontinued = false
GROUP BY p.product_id, p.product_name, p.reorder_point, p.unit_cost
HAVING SUM(i.quantity_on_hand) <= p.reorder_point
ORDER BY units_to_reorder DESC;
""",
    },
]


@tool
def load_skill(skill_name: str) -> str:
    """将指定技能的完整内容加载到智能体的上下文中。

    当你需要关于如何处理某类请求的详细信息时，请使用此工具。
    它会为你提供该技能领域的全面说明、策略和指南。

    参数:
        skill_name: 要加载的技能名称（例如："sales_analytics"、"inventory_management"）
    """
    # 查找并返回请求的技能
    for skill in SKILLS:
        if skill["name"] == skill_name:
            return f"已加载技能：{skill_name}\n\n{skill['content']}"

    # 未找到技能
    available = ", ".join(s["name"] for s in SKILLS)
    return f"未找到技能 '{skill_name}'。可用技能有：{available}"


@tool
def run_shell(command: str):
    """
    运行给定的shell命令
    :param command: 待运行的shell命令
    :return: shell命令返回结果
    """
    pass


def get_model():
    base_url = "https://gateway.ai.cloudflare.com/v1/67b8ebfcb6b836e009e1fb540f160fa5/nlp_0314/openrouter/v1"
    api_key = "sk-or-v1-"
    model_name = "qwen/qwen3-235b-a22b-2507"
    max_tokens = None

    return ChatOpenAI(
        streaming=True,
        verbose=True,
        callbacks=None,
        openai_api_key=api_key,
        openai_api_base=base_url,
        model_name=model_name,
        temperature=0.9,
        max_tokens=max_tokens
    )


class SkillMiddleware(AgentMiddleware):
    """将技能描述注入系统提示（system prompt）的中间件。"""

    # 将 load_skill 工具注册为类变量
    tools = [load_skill]

    def __init__(self):
        """初始化并根据 SKILLS 列表生成技能提示文本。"""
        # 从 SKILLS 列表构建技能提示
        skills_list = []
        for skill in SKILLS:
            skills_list.append(
                f"- **{skill['name']}**: {skill['description']}"
            )
        self.skills_prompt = "\n".join(skills_list)

    def wrap_model_call(
            self,
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """同步方法：将技能描述注入系统提示中。"""
        # 构建技能附加说明
        skills_addendum = (
            f"\n\n## 可用技能\n\n{self.skills_prompt}\n\n"
            "当你需要关于处理特定类型请求的详细信息时，请使用 load_skill 工具。"
        )

        # 将附加说明追加到系统消息的内容块中
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)


if __name__ == '__main__':
    model = get_model()

    # 创建agent
    agent = create_agent(
        model=model,
        system_prompt=(
            "你是一个 SQL 查询助手，帮助用户编写针对业务数据库的查询语句。"
        ),
        middleware=[SkillMiddleware()],
        checkpointer=InMemorySaver()
    )

    while True:
        query = input("我:").strip()
        if query == 'q':
            break
        if len(query) == 0:
            continue

        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": query
                    }
                ]
            },
            {
                "run_id": str(uuid.uuid4()),
                "configurable": {
                    "thread_id": str(uuid.uuid4())
                }
            }
        )
        for msg in result['messages']:
            if hasattr(msg, 'pretty_print'):
                msg.pretty_print()
            else:
                print(f"{msg.type}: {msg.content}")

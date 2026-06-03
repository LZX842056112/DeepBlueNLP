# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/21 15:32
Create User : 19410
Desc : 产品查询相关的工具方法
"""

import logging
from typing import Dict, Any

import requests
from mcp.server import FastMCP

logging.basicConfig(
    level=logging.INFO
)
mcp = FastMCP("weather", port=8000, host="0.0.0.0")


@mcp.tool()
def product_order_number(product_id: int) -> dict:
    """
    获取产品对应的订单数目
    :param product_id: 产品id
    :return: 订单数目
    """
    logging.info(f"本地商品订单数目查询:{product_id}")
    sid = int(product_id)
    if sid == 1:
        return {"订单数目": 100}
    else:
        return {"订单数目": 200 + sid * 5}


@mcp.tool()
def product_price(product_id) -> dict:
    """
    获取产品对应的单价
    :param product_id: 产品id
    :return: 单价信息
    """
    print(f"本地订单商品单价查询:{product_id}")
    sid = int(product_id)
    if sid == 1:
        return {"价格": 35.25}
    else:
        return {"价格": 36.2}


if __name__ == '__main__':
    logging.info("start product mcp server")
    mcp.run(transport="streamable-http")

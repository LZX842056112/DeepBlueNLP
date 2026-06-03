# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/22 10:12
Create User : 19410
Desc : 提供数学计算相关的MCP服务
"""
import logging

from mcp.server import FastMCP

logging.basicConfig(
    level=logging.INFO
)
mcp = FastMCP("Math")


@mcp.tool()
def add(a: float, b: float) -> float:
    """
    计算两个数的和
    :param a: 加数1
    :param b: 加数2
    :return: 结果
    """
    logging.info(f"计算两个数的和 {a} {b}")
    return a + b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """
    计算两个数的积
    :param a: 乘数1
    :param b: 乘数2
    :return: 乘机
    """
    logging.info(f"计算两个数的积 {a} {b}")
    return a * b


if __name__ == '__main__':
    logging.info("start math mcp server")
    mcp.run(transport="stdio")

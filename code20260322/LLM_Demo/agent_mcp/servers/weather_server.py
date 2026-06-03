# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/22 10:36
Create User : 19410
Desc : 天气的MCP服务器
"""
import logging
from typing import Dict, Any

import requests
from mcp.server import FastMCP

logging.basicConfig(
    level=logging.INFO
)
mcp = FastMCP("weather")


def weather(location: str, api_key: str) -> Dict[str, Any]:
    url = f"https://api.seniverse.com/v3/weather/now.json?key={api_key}&location={location}&language=zh-Hans&unit=c"
    logging.info(f"get city weather {url}")
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


@mcp.tool()
def weathercheck(location: str) -> Dict[str, Any]:
    """
    获取对应地区的天气情况
    :param location: 城市名称或者地区名称
    :return: 天气情况字典信息
    """
    return weather(location, '')


if __name__ == '__main__':
    logging.info("start weather mcp server")
    mcp.run(transport="stdio")

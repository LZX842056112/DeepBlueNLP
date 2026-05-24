# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/21 16:38
Create User : 19410
Desc : 数学工具方法
"""


from pydantic import BaseModel, Field


def add(a, b):
    print(f"计算加法:{a} + {b}")
    return float(a) + float(b)


class MathAddInput(BaseModel):
    a: float = Field(description="第一个加数")
    b: float = Field(description="第二个加数")



def mul(a, b):
    print(f"计算乘法:{a} * {b}")
    return float(a) * float(b)


class MathMulInput(BaseModel):
    a: float = Field(description="第一个乘数")
    b: float = Field(description="第二个乘数")



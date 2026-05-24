# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/21 15:32
Create User : 19410
Desc : 产品查询相关的工具方法
"""

from pydantic import BaseModel, Field


def product_order_number(product_id):
    """
    模拟过程，正常逻辑 -> 调用后端程序开发的api接口
    :param sid:
    :return:
    """
    print(f"本地商品订单数目查询:{product_id}")
    sid = int(product_id)
    if sid == 1:
        return {"订单数目": 100}
    else:
        return {"订单数目": 200 + sid * 5}


class ProductOrderNumberInput(BaseModel):
    product_id: int = Field(description="商品id")


def product_price(product_id):
    print(f"本地订单商品单价查询:{product_id}")
    sid = int(product_id)
    if sid == 1:
        return {"价格": 35.25}
    else:
        return {"价格": 36.2}


class ProductPriceInput(BaseModel):
    product_id: int = Field(description="商品id")

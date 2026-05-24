from tools import *
from tools.multi_args_tool import SupportDictArgsTool

tools = [
    SupportDictArgsTool.from_function(
        func=weathercheck,  # 对应的执行函数
        name="weather_check",  # 方法名称 --> 一般会传递给LLM
        description="获取当前对应城市的天气，需要传入城市名称",  # 方法的描述 --> 一般会传递给LLM
        args_schema=WeatherInput,  # 方法的入参格式 -- 一般传递给LLM
    ),
    SupportDictArgsTool.from_function(
        func=weathercheck_with_days,
        name="weathercheck_with_days",
        description="获取未来几天对应城市的天气，需要传入城市名称和天数",
        args_schema=WeatherInputWithDays,
    ),
    SupportDictArgsTool.from_function(
        func=product_order_number,
        name="product_order_number",
        description="基于给定的商品id，从数据库中获取当前商品的订单总数目，需要传入商品id",
        args_schema=ProductOrderNumberInput,
    ),
    SupportDictArgsTool.from_function(
        func=product_price,
        name="product_price",
        description="支持基于给定商品id获取该商品的单价，需要传入商品id",
        args_schema=ProductPriceInput,
    ),
    SupportDictArgsTool.from_function(
        func=search_internet,
        name="search_internet",
        description="使用该工具可以访问互联网获取联网搜索结果，需要传入搜索关键词",
        args_schema=SearchInternetInput,
    ),
    SupportDictArgsTool.from_function(
        func=add,
        name="add",
        description="计算两个数的和",
        args_schema=MathAddInput,
    ),
    SupportDictArgsTool.from_function(
        func=mul,
        name="mul",
        description="计算两个数的乘积",
        args_schema=MathMulInput,
    )
]

tool_names = [tool.name for tool in tools]

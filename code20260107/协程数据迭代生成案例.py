# -*- coding: utf-8 -*-


import asyncio
import random
from typing import AsyncGenerator
import threading


async def async_data_generator(cnt: int) -> AsyncGenerator[int, None]:
    thread = threading.current_thread()
    print(f"async_data_generator ----> {thread.name}_{thread.ident}")
    for i in range(cnt):
        # 等待 --> 模拟异步操作
        await asyncio.sleep(random.uniform(0.5, 1.5))
        # 产生随机数
        rnd_data = i * 10 + random.randint(1, 9)
        print(f"数据生成:{rnd_data}")
        # 数据返回
        yield rnd_data


async def main():
    thread = threading.current_thread()
    print(f"main ----> {thread.name}_{thread.ident}")
    data_iter = async_data_generator(5)
    print(f"迭代器对象:{data_iter}")

    print("开始获取迭代器内的数据.....")
    async for data in data_iter:
        print(f"数据:{thread.name}_{thread.ident} ---> {data}")


if __name__ == '__main__':
    asyncio.run(main())

# -*- coding: utf-8 -*-
import asyncio
import os
import threading
import datetime
import time


async def coro_task(task_name, delay):
    """协程任务：模拟耗时 I/O 操作"""
    _msg = f"进程id:{os.getpid()} 父进程id:{os.getppid()} 线程id:{threading.current_thread().ident}"
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')}] {_msg} 协程任务 {task_name} 开始执行，延迟 {delay} 秒")

    # 异步睡眠（非阻塞，期间事件循环调度其他协程）
    await asyncio.sleep(delay)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')}] {_msg} 协程任务 {task_name} 执行完成")

    # 可直接返回结果
    return f"{task_name} {_msg} 结果"


async def main_coro():
    """协程主函数"""
    print("===== 协程执行开始 =====")
    start_time = time.time()

    # 1. 并发执行多个协程任务（两种方式任选）
    # 方式1：create_task 手动创建任务
    task1 = asyncio.create_task(coro_task("任务A", 20), name="Coro-1")
    task2 = asyncio.create_task(coro_task("任务B", 10), name="Coro-2")
    task3 = asyncio.create_task(coro_task("任务C", 30), name="Coro-3")

    # 等待所有任务完成，获取结果
    result1 = await task1
    result2 = await task2
    result3 = await task3

    # 方式2：gather 批量执行（更简洁，推荐批量任务）
    # results = await asyncio.gather(
    #     coro_task("任务A", 2),
    #     coro_task("任务B", 1),
    #     coro_task("任务C", 3)
    # )

    end_time = time.time()
    print(f"===== 协程执行结束 =====")
    print(f"任务结果：{result1}, {result2}, {result3}")
    print(f"总耗时：{end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    # 启动协程程序（自动管理事件循环）
    asyncio.run(main_coro())

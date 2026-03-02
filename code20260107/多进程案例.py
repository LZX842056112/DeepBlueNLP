# -*- coding: utf-8 -*-
import datetime
import multiprocessing
import os
import threading
import time


# noinspection DuplicatedCode
def process_task(task_name, delay, result_queue):
    """进程任务：模拟耗时操作（CPU 密集型/ I/O 密集型）"""
    _msg = (f"进程id:{os.getpid()} 父进程id:{os.getppid()} "
            f"线程id:{threading.current_thread().ident} 线程名称:{threading.current_thread().name}")
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')}] {_msg} 进程任务 {task_name} 开始执行，延迟 {delay} 秒")

    # 模拟耗时（CPU 密集型用循环，I/O 密集型用time.sleep，此处统一用sleep）
    time.sleep(delay)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')}] {_msg} 进程任务 {task_name} 执行完成")

    # 将结果放入队列（进程间数据共享方式之一）
    result = f"{task_name} {_msg} 结果"
    result_queue.put(result)


def main_process():
    """多进程主函数"""
    _msg = (f"进程id:{os.getpid()} 父进程id:{os.getppid()} "
            f"线程id:{threading.current_thread().ident} 线程名称:{threading.current_thread().name}")
    print("===== 多进程执行开始 =====")
    print(_msg)
    start_time = time.time()

    # 1. 创建结果队列（用于进程间传递结果，IPC 机制）
    result_queue = multiprocessing.Queue()

    # 2. 创建进程对象
    process1 = multiprocessing.Process(
        target=process_task,
        args=("任务A", 20, result_queue),
        name="Process-1"
    )
    process2 = multiprocessing.Process(
        target=process_task,
        args=("任务B", 10, result_queue),
        name="Process-2"
    )
    process3 = multiprocessing.Process(
        target=process_task,
        args=("任务C", 30, result_queue),
        name="Process-3"
    )

    # 3. 启动所有进程
    process1.start()
    process2.start()
    process3.start()

    # 4. 等待所有进程执行完成
    process1.join()
    process2.join()
    process3.join()

    # 5. 从队列中获取所有结果
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    print(f"所有任务结果：{results}")

    end_time = time.time()
    print(f"===== 多进程执行结束 =====")
    print(f"总耗时：{end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    main_process()

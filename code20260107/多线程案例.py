# -*- coding: utf-8 -*-
"""
threading demo
"""
import datetime
import threading
import time
import os


def thread_task(task_name, delay):
    """线程任务：模拟耗时操作（I/O 密集型）"""
    _msg = (f"进程id:{os.getpid()} 父进程id:{os.getppid()} "
            f"线程id:{threading.current_thread().ident} 线程名称:{threading.current_thread().name}")
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')}] {_msg} 线程任务 {task_name} 开始执行，延迟 {delay} 秒")

    # 模拟耗时（此处用time.sleep模拟I/O阻塞，线程会释放GIL，允许其他线程执行）
    time.sleep(delay)
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')}] {_msg} 线程任务 {task_name} 执行完成")

    # threading库 多线程模块中默认不支持返回结果，需要通过其它方式来获取返回结果
    return f"{task_name} {_msg} 结果"


def main_thread():
    """多线程主函s数"""
    print("===== 多线程执行开始 =====")
    start_time = time.time()

    # 1. 创建线程对象
    thread1 = threading.Thread(target=thread_task, args=("任务A", 20), name="Thread-1")
    thread2 = threading.Thread(target=thread_task, args=("任务B", 10), name="Thread-2")
    thread3 = threading.Thread(target=thread_task, args=("任务C", 30), name="Thread-3")

    # 2. 启动所有线程（启动后线程进入就绪状态，由系统调度执行）
    thread1.start()
    thread2.start()
    thread3.start()

    # 3. 等待所有线程执行完成（主线程阻塞，直到子线程全部结束）
    thread1.join()
    thread2.join()
    thread3.join()

    end_time = time.time()
    print(f"===== 多线程执行结束 =====")
    print(f"总耗时：{end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    main_thread()

# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/18 20:12
Create User : 19410
Desc : 数据得到简单处理
"""

import os

input_folder = r"./西游记原文"
output_folder = r"./output/原文"

# 创建文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历数据
cnt = 0
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        # 构建文件路径
        input_file_path = os.path.join(input_folder, filename)

        # 读取文件内容
        with open(input_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 去除每行的前后空格
        lines = [line.strip() for line in lines]

        # 获取新的文件名（原文件的第一行）
        new_filename = lines[0] + '.txt'
        output_file_path = os.path.join(output_folder, new_filename)

        # 将修改后的内容写入输出文件
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(lines[1:]).strip() + '\n')
        cnt += 1

print(f"总保存的文件数目为:{cnt}")

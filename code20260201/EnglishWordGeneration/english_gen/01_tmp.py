# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/30 20:16
Create User : 19410
Desc : 数据处理：将原始数据转换为单词
"""

import os

if __name__ == '__main__':

    in_file = "./datas/text8"
    out_file = "./datas/words.txt"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    total_words = set()
    with open(in_file, 'r', encoding='utf-8') as reader:
        for line in reader:
            words = set(line.split(" "))
            for word in words:
                word = word.lower()
                if word.isalpha():  # 仅添加只有英文字母的单词
                    total_words.add(word)
    print(f"总单词数目:{len(total_words)}")

    total_words = list(total_words)
    total_words = sorted(total_words)
    cnt = 0
    with open(out_file, 'w', encoding="utf-8") as writer:
        for word in total_words:
            if len(word) > 2:
                writer.writelines(f'{word}\n')
                cnt += 1
    print(f"总有效单词数目为:{cnt}")

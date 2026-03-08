# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/30 20:20
Create User : 19410
Desc : 数据加载器：构造数据加载器，最终返回模型需要的数据格式
PS: 用上一个token的信息去预测下一个token是什么
{
    "input_ids": torch.tensor([[2,14,11,18,18,22,3],......]),
    "labels": torch.tensor([[2,14,11,18,18,22,3],......]),
    "attention_mask": torch.tensor([[1,1,1,1,1,1,1],....]),
    "text": ["[CLS] hello [SEP]",....]
}
"""
import copy

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer


class MyDataset(Dataset):
    def __init__(self, word_file, tokenizer: BertTokenizer):
        super(MyDataset, self).__init__()
        self.tokenizer = tokenizer
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.datas = self.load_data(word_file)

    def load_data(self, word_file):
        datas = []
        with open(word_file, 'r', encoding='utf-8') as reader:
            for word in reader:
                word = word.strip().lower()
                tokens = list(word)  # 将每个字母看成一个一个的输入token
                # 如果已经分好词，当前仅需要进行token id转换，调用encode_plus；否则调用__call__方法
                input_ids = self.tokenizer.encode_plus(tokens)['input_ids']
                datas.append((input_ids, f"{self.cls_token} {word} {self.sep_token}", len(input_ids)))
        return datas

    def __getitem__(self, index):
        input_ids, text, input_token_num = self.datas[index]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "text": text,
            "attention_mask": torch.ones(input_token_num, dtype=torch.int64)
        }

    def __len__(self):
        return len(self.datas)


def padding_tensors(values):
    return torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=0)


def collate_fn(batch):
    result = {}
    keys = batch[0].keys()
    bs = len(batch)
    for key in keys:
        values = [batch[i][key] for i in range(bs)]
        if isinstance(values[0], torch.Tensor):
            values = padding_tensors(values)  # 数据填充，均填充0
        result[key] = values

    if 'labels' not in result:
        labels = copy.deepcopy(result['input_ids'])
        labels[labels == 0] = -100  # 填充位置不计算损失
        result['labels'] = labels

    return result


def t0():
    tokenizer = BertTokenizer(
        vocab_file="datas/vocab.txt",
        do_lower_case=True
    )
    ds = MyDataset(
        word_file="datas/words.txt",
        tokenizer=tokenizer
    )
    print(ds[75])

    dataloader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    for batch in dataloader:
        print(batch)
        break


if __name__ == '__main__':
    t0()

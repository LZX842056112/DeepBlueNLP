# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/30 20:37
Create User : 19410
Desc : 基于LSTM模型结构的训练
"""

import copy
import os.path

import torch
import torch.nn as nn
from torch import optim
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


class Network(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Network, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False,  # 一定为False
            num_layers=3
        )
        self.classify = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask, labels=None):
        """
        :param input_ids: [N,T]
        :param attention_mask: [N,T]
        :param labels: [N,T]
        :return:
        """
        # 1. 获取输入token对应的特征向量 [N,T,E]
        feature_embed = self.embedding_layer(input_ids)

        # 2. 进一步的提取特征向量 [N,T,E]
        feature_embed, _ = self.lstm(feature_embed)

        # 3. 针对每个token进行分类决策判断，得到置信度 [N,T,vocab_size]
        logits = self.classify(feature_embed)

        # 4. 如果给定标签的情况下，进行损失计算
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = self.loss_fn(shift_logits.reshape(-1, self.vocab_size), shift_labels.reshape(-1))
        return logits, loss


def training(output_dir, vocab_file, word_file, batch_size, hidden_size, total_epoch):
    model_file = os.path.join(output_dir, "model", "net.pkl")
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"当前运行环境为:{device}")

    # 数据加载相关代码逻辑
    tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=True)
    ds = MyDataset(word_file=word_file, tokenizer=tokenizer)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 模型创建
    net = Network(vocab_size=tokenizer.vocab_size, hidden_size=hidden_size)
    net.to(device=device)
    opt = optim.AdamW(params=net.parameters(), lr=0.001)

    # 遍历数据迭代训练
    for epoch in range(total_epoch):
        # 训练
        net.train()
        for batch in dataloader:
            # 将tensor转换到对应的设备上
            for key in batch.keys():
                value = batch[key]
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device=device)

            # 前向过程
            logits, loss = net(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )

            # 反向传播
            loss.backward()
            opt.step()
            opt.zero_grad()

            # 训练过程中的日志输出
            print(f"{epoch} Loss: {loss.item():.3f}")

        # 评估

        # 持久化
        torch.save(net.eval(), model_file)

if __name__ == '__main__':
    training(
        output_dir="output/04_lstm",
        vocab_file="datas/vocab.txt",
        # word_file="./datas/words.txt",
        word_file="datas/words_min.txt",
        batch_size=8,
        hidden_size=64,
        total_epoch=100
    )
# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/30 20:37
Create User : 19410
Desc : 基于Bert模型的训练
"""

import copy
import os.path

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertConfig, BertLMHeadModel


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


def training(output_dir, vocab_file, word_file, batch_size, hidden_size, total_epoch):
    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"当前运行环境为:{device}")

    # 数据加载相关代码逻辑
    tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=True)
    tokenizer.save_pretrained(model_dir)  # 先保存一下
    ds = MyDataset(word_file=word_file, tokenizer=tokenizer)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 模型创建
    cfg = BertConfig(
        is_decoder=True,
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=hidden_size * 4,
    )
    net = BertLMHeadModel(cfg)
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
            net_output = net(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                use_cache=False,
                return_dict=True
            )
            loss = net_output.loss

            # 反向传播
            loss.backward()
            opt.step()
            opt.zero_grad()

            # 训练过程中的日志输出
            print(f"{epoch} Loss: {loss.item():.3f}")

        # 评估

        # 持久化
        net.save_pretrained(model_dir)


@torch.no_grad()
def predict():
    output_dir = "./output/06_bert"
    model_dir = os.path.join(output_dir, "model")
    tokenizer = BertTokenizer.from_pretrained(model_dir)  # 恢复解析器
    bert = BertLMHeadModel.from_pretrained(model_dir)  # 恢复模型
    bert.eval()

    def _predict(_input_ids):
        output_token_ids = copy.deepcopy(_input_ids)
        new_tokens = 0
        past_key_values = None
        while new_tokens < 10:
            bert_output = bert(
                _input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            past_key_values = bert_output.past_key_values

            logits = bert_output.logits
            logits = logits[:, -1, :]  # 最后一个时刻的预测置信度
            # 选择当前时刻预测概率最大的token id
            pred_ids = torch.argmax(logits, dim=-1, keepdim=True)  # [N,1]

            if pred_ids[0, 0].item() == tokenizer.sep_token_id:
                break
            output_token_ids = torch.concat([output_token_ids, pred_ids], dim=1)
            _input_ids = pred_ids

            new_tokens += 1
        # 将id转换为token
        t = tokenizer.convert_ids_to_tokens(list(output_token_ids.cpu().numpy()[0]))
        return "".join(t[1:])

    while True:
        text = input("请输入单词前缀:")
        if text == '1':
            break
        input_ids = tokenizer.encode_plus(list(text))['input_ids'][:-1]
        word = _predict(torch.tensor([input_ids], dtype=torch.int64))
        print(word)


@torch.no_grad()
def predict_with_generate():
    output_dir = "./output/06_bert"
    model_dir = os.path.join(output_dir, "model")
    tokenizer = BertTokenizer.from_pretrained(model_dir)  # 恢复解析器
    bert = BertLMHeadModel.from_pretrained(model_dir)  # 恢复模型
    bert.eval()

    def _predict(_input_ids):
        # generate方法的内部类似我们自己写的predict方法
        # NOTE: generate生成方法更偏向于生成长序列的文本，如何生成短文本呢？
        # output_token_ids = bert.generate(_input_ids, eos_token_id=tokenizer.sep_token_id)
        output_token_ids = bert.generate(_input_ids, top_k=5, do_sample=True, eos_token_id=tokenizer.sep_token_id)
        # 将id转换为token
        t = tokenizer.convert_ids_to_tokens(list(output_token_ids.cpu().numpy()[0]))
        return "".join(t[1:])

    while True:
        text = input("请输入单词前缀:")
        if text == '1':
            break
        input_ids = tokenizer.encode_plus(list(text))['input_ids'][:-1]
        word = _predict(torch.tensor([input_ids], dtype=torch.int64))
        print(word)


if __name__ == '__main__':
    # training(
    #     output_dir="./output/06_bert",
    #     vocab_file="./datas/vocab.txt",
    #     # word_file="./datas/words.txt",
    #     word_file="./datas/words_min.txt",
    #     batch_size=8,
    #     hidden_size=128,
    #     total_epoch=100
    # )
    # predict()
    predict_with_generate()

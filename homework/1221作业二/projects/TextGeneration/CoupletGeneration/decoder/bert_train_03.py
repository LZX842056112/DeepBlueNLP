# -*- coding: utf-8 -*-

import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
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
            for line in reader:
                line = line.strip()
                text1, text2 = line.split("，")  # 分割为上联和下联
                token1 = list(text1)  # 上联分词
                token2 = list(text2)  # 下联分词

                # 需求一数据构造
                tokens = list("下联生成:") + token1 + ["，"] + token2
                input_ids = self.tokenizer.encode_plus(tokens)['input_ids']
                datas.append((input_ids, f"{self.cls_token} 下联生成:{line} {self.sep_token}", len(input_ids)))

                # 需求2数据构造
                prefix = f"上下联生成:{token1[0]}{token2[0]};"
                tokens = list(prefix) + token1 + ["，"] + token2
                input_ids = self.tokenizer.encode_plus(tokens)['input_ids']
                datas.append((input_ids, f"{self.cls_token} {prefix}{line} {self.sep_token}", len(input_ids)))
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
        # noinspection PyUnresolvedReferences
        labels = copy.deepcopy(result['input_ids'])
        labels[labels == 0] = -100  # 填充位置不计算损失
        result['labels'] = labels

    return result


def train(vocab_file, word_file, output_dir, hidden_size=128, batch_size=8, total_epoch=100):
    model_save_dir = os.path.join(output_dir, "model")
    os.makedirs(model_save_dir, exist_ok=True)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # 数据加载相关代码逻辑
    tokenizer = BertTokenizer(vocab_file=vocab_file, do_lower_case=True)
    tokenizer.save_pretrained(model_save_dir)  # 先保存一下
    ds = MyDataset(word_file=word_file, tokenizer=tokenizer)
    dataloader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

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

    # 遍历数据进行训练
    opt.zero_grad()
    for epoch in range(total_epoch):
        net.train()
        for batch in dataloader:
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

            print(f"{epoch} Loss: {loss.item():.3f}")

    # 模型持久化保存
    net.save_pretrained(model_save_dir)


@torch.no_grad()
def predict():
    output_dir = "output/bert03"
    model_save_dir = os.path.join(output_dir, "model")
    tokenizer = BertTokenizer.from_pretrained(model_save_dir)  # 恢复解析器
    bert = BertLMHeadModel.from_pretrained(model_save_dir)  # 恢复模型
    bert.eval()

    def _predict(_input_ids):
        output_token_ids = copy.deepcopy(_input_ids)
        new_tokens = 0
        past_key_values = None
        while new_tokens < 30:
            bert_output = bert(
                _input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            past_key_values = bert_output.past_key_values

            logits = bert_output.logits
            logits = logits[:, -1, :]  # 最后一个时刻的预测置信度
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
        k = 0
        text = input("请输入两个字或者五个字:")
        if text.isnumeric():
            k = int(text)
            text = input("请再次输入两个字或者五个字:")

        if k == 2 or (k == 0 and len(text) == 2):
            text = f"上下联生成:{text};"
        elif 5 == k or (k == 0 and len(text) == 5):
            text = f"下联生成:{text}，"
        else:
            print(f"当前text文本异常:{text}")
            text = None

        if text is not None:
            print(f"原始文本:{text}")
            input_ids = tokenizer.encode_plus(list(text))['input_ids'][:-1]
            result = _predict(torch.tensor([input_ids], dtype=torch.int64))
            print(result[len(text):])


@torch.no_grad()
def predict_with_generate():
    output_dir = "output/bert03"
    model_save_dir = os.path.join(output_dir, "model")
    tokenizer = BertTokenizer.from_pretrained(model_save_dir)  # 恢复解析器
    bert = BertLMHeadModel.from_pretrained(model_save_dir)  # 恢复模型
    bert.eval()

    def _predict(_input_ids):
        # generate方法的内部类似我们自己写的predict方法
        # NOTE: generate生成方法更偏向于生成长序列的文本，如何生成短文本呢？
        output_token_ids = bert.generate(
            _input_ids, eos_token_id=tokenizer.sep_token_id, max_new_tokens=14
        )
        # 将id转换为token
        t = tokenizer.convert_ids_to_tokens(list(output_token_ids.cpu().numpy()[0]))
        return "".join(t[1:])

    while True:
        text = input("请输入两个字或者五个字:")
        if len(text) == 2:
            text = f"上下联生成:{text};"
        elif len(text) == 5:
            text = f"下联生成:{text}，"
        else:
            print(f"当前text文本异常:{text}")
            text = None

        if text is not None:
            input_ids = tokenizer.encode_plus(list(text))['input_ids'][:-1]
            word = _predict(torch.tensor([input_ids], dtype=torch.int64))
            print(word)
            print(word[len(text):])


if __name__ == '__main__':
    train(
        vocab_file="../datas/vocab.txt",
        word_file="../datas/poetry_min.txt",
        output_dir="output/bert03",
        hidden_size=128, batch_size=4, total_epoch=100
    )
    # predict()
    predict_with_generate()

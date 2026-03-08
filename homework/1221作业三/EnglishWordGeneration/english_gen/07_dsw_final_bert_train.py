# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/30 21:10
Create User : 19410
Desc : DSW上训练的一个bert模型
"""

import copy
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
if 'nt' in os.name:
    os.environ['XDG_CACHE_HOME'] = r'D:/cache'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

try:
    import torchmetrics  # pip install torchmetrics==1.4.0
except:
    torchmetrics = None
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertLMHeadModel, GenerationConfig


class MyDataset(Dataset):
    def __init__(self, word_file, tokenizer: BertTokenizer):
        super(MyDataset, self).__init__()
        self.tokenizer = tokenizer
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.datas = self.load_data(word_file)

    def load_data(self, word_file):
        datas = []
        with open(word_file, "r", encoding="utf-8") as reader:
            for word in tqdm(reader):
                word = word.strip().lower()
                chars = list(word)
                input_ids = self.tokenizer.encode_plus(chars)['input_ids']
                datas.append((f"{self.cls_token} {word} {self.sep_token}", input_ids, len(input_ids)))

                # if len(datas) > 2000:
                #     break
        return datas

    def __getitem__(self, index):
        text, input_ids, input_token_num = self.datas[index]
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.int64),
            'text': text,
            'attention_mask': torch.ones(input_token_num, dtype=torch.int64)
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
            values = padding_tensors(values)

        result[key] = values

    if 'labels' not in result:
        labels = copy.deepcopy(result['input_ids'])
        labels[labels == 0] = -100  # 填充位置全部重置为-100，不参与计算loss计算
        result['labels'] = labels

    return result


def train():
    output_dir = "output/final"
    model_save_dir = os.path.join(output_dir, "model")
    summary_save_dir = os.path.join(output_dir, "logs")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(summary_save_dir, exist_ok=True)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # 1. 数据加载
    tokenizer = BertTokenizer(vocab_file='datas/vocab.txt', do_lower_case=True)
    tokenizer.save_pretrained(model_save_dir)
    vocab_size = tokenizer.vocab_size
    ds = MyDataset(
        # word_file=r"./datas/words.txt",
        word_file=r"datas/words_min.txt",
        tokenizer=tokenizer
    )
    dataloader = DataLoader(
        ds,
        batch_size=10,  # 生成类型的模型一般批次都是非常大的
        shuffle=True,
        collate_fn=collate_fn
    )

    # 2. 模型创建
    cfg = BertConfig(
        is_decoder=True,
        vocab_size=vocab_size,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        intermediate_size=256 * 4,
        use_cache=True
    )
    bert = BertLMHeadModel(cfg)
    bert = bert.to(device=device)

    # 3. 优化器、损失函数
    opt = optim.AdamW(params=bert.parameters(), lr=0.001)
    if torchmetrics is not None:
        acc_top1 = torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size, top_k=1, ignore_index=-100)
        acc_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size, top_k=5, ignore_index=-100)
    writer = SummaryWriter(log_dir=summary_save_dir)

    # 4. 遍历数据进行训练
    global_step = 0
    opt.zero_grad()
    for epoch in range(2000):
        bert.train()
        for batch in dataloader:
            for key in batch.keys():
                value = batch[key]
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device=device)

            # 1. 前向过程
            bert_output = bert(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                use_cache=False,
                return_dict=True
            )
            loss = bert_output.loss
            logits = bert_output.logits
            preds = torch.softmax(logits, dim=-1)
            preds = preds * batch['attention_mask'][..., None]
            preds[:, :, 0] += 1.0 - batch['attention_mask']
            preds = preds[:, :-1, :].cpu().contiguous()

            # 2. 梯度反向更新
            loss.backward()
            opt.step()
            opt.zero_grad()

            labels = batch['labels']
            labels = labels[:, 1:].cpu().contiguous()
            if torchmetrics is not None:
                _acc1 = acc_top1(preds.view(-1, vocab_size), labels.view(-1))
                _acc2 = acc_top5(preds.view(-1, vocab_size), labels.view(-1))
                print(f"{epoch} Loss: {loss.item():.3f} Acc1:{_acc1:.5f} Acc2:{_acc2:.5f}")
                writer.add_scalar('train_top1_acc', _acc1.item(), global_step=global_step)
                writer.add_scalar('train_top5_acc', _acc2.item(), global_step=global_step)
            else:
                print(f"{epoch} Loss: {loss.item():.3f}")
            writer.add_scalar('train_loss', loss.item(), global_step=global_step)
            global_step += 1

        if epoch % 10 == 0:
            bert.save_pretrained(model_save_dir)

    bert.save_pretrained(model_save_dir)
    writer.close()


@torch.no_grad()
def predict():
    output_dir = "output/final"
    model_save_dir = os.path.join(output_dir, "model")
    tokenizer = BertTokenizer.from_pretrained(model_save_dir)  # 恢复解析器
    bert = BertLMHeadModel.from_pretrained(model_save_dir)  # 恢复模型
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
    output_dir = "output/final"
    model_save_dir = os.path.join(output_dir, "model")
    tokenizer = BertTokenizer.from_pretrained(model_save_dir)  # 恢复解析器
    bert = BertLMHeadModel.from_pretrained(model_save_dir)  # 恢复模型
    bert.eval()

    def _predict(_input_ids):
        # generate方法的内部类似我们自己写的predict方法
        # NOTE: generate生成方法更偏向于生成长序列的文本，如何生成短文本呢？

        # 控制生成逻辑的相关配置信息
        generation_config = GenerationConfig(
            # guidance_scale=0.8,
            max_new_tokens=20,  # 最多允许生成20个新token
            temperature=1.8,  # 温度系数 --> 更改模型的输出置信度
            top_k=4,  # 当进行sample采样的时候，仅保留多少个最大置信度的预测token
            # top_p=0.7,  # 范围系数，所有置信度大的token预测概率累计值不能超过top_p
            do_sample=True, num_beams=1  # sample的主要参数
            # num_beams=4, do_sample=False, num_beam_groups=1  # beam search的主要参数
            # num_beams=2, do_sample=True, num_beam_groups=1  # beam sample的主要参数
            # num_beams=4, do_sample=False, num_beam_groups=2, diversity_penalty=0.1  # group beam search的主要参数
            # num_beams=1, do_sample=False, penalty_alpha=0.2  # top_k>1, contrastive_search的主要参数
        )

        # output_token_ids = bert.generate(
        #     _input_ids, eos_token_id=tokenizer.sep_token_id,
        #     do_sample=True, num_beams=10,
        #     early_stopping=True, temperature=1.2
        # )
        output_token_ids = bert.generate(
            _input_ids,
            generation_config=generation_config,
            eos_token_id=tokenizer.sep_token_id,
            num_return_sequences=2
        )
        print(output_token_ids)
        # 将id转换为token
        bs = output_token_ids.shape[0]
        result = []
        for i in range(bs):
            t = tokenizer.convert_ids_to_tokens(list(output_token_ids[i].cpu().numpy()))
            t = "".join(t[1:]).replace("[SEP]", "").replace("[PAD]", "")
            result.append(t)
        return result

    while True:
        text = input("请输入单词前缀:")
        if text == '1':
            break
        input_ids = tokenizer.encode_plus(list(text))['input_ids'][:-1]
        word = _predict(torch.tensor([input_ids], dtype=torch.int64))
        print(word)


if __name__ == '__main__':
    # train()
    # predict()
    predict_with_generate()

# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/24 21:27
Create User : 19410
Desc : 训练的入口类
"""
import os
from datetime import datetime

import pandas as pd
import torch
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..config import Config
from ..datas.dataset import TextClassifyDataset
from ..datas.utils import build_dataloader
from ..models.common import LSTMTextClassifyNetwork
from ..utils import build_losses, build_optim


class Trainer(object):
    def __init__(self, config: Config):
        super().__init__()

        self.train_batch_steps = 0
        self.test_batch_steps = 0
        self.config = config
        self.summary_dir = self.config.summary_dir
        self.total_epoch = self.config.total_epoch
        os.makedirs(self.config.model_output_dir, exist_ok=True)
        self.last_model_path = os.path.join(self.config.model_output_dir, "last.pkl")

        # 1. 分词器的构建
        self.tokenizer = self.config.tokenizer

        # 2. 训练数据构造
        self.train_dataset, self.train_dataloader = self.load_train_dataloader()

        # 3. 验证数据的加载
        self.val_dataset, self.val_dataloader = self.load_eval_dataloader()

        # 4. 网络结构创建
        self.net = self.load_network()

        # 损失函数创建
        self.loss_fn = build_losses()

        # 优化器创建
        self.opt = build_optim(self.net, lr=self.config.lr)

        # 可视化日志输出对象的构建
        self.writer = self.load_summary_writer()

    def load_train_dataloader(self):
        return self.load_dataloader(
            data_file=self.config.train_file,
            batch_size=self.config.batch_size,
            shuffle=True
        )

    def load_eval_dataloader(self):
        return self.load_dataloader(
            data_file=self.config.eval_file,
            batch_size=self.config.batch_size * 2,
            shuffle=False
        )

    def load_dataloader(self, data_file, batch_size, shuffle=True):
        df = pd.read_csv(data_file, sep="\t", header=None, names=['text', 'label'])
        ds = TextClassifyDataset(
            texts=df.text.values,
            labels=df.label.values,
            tokenizer=self.tokenizer
        )
        dataloader = build_dataloader(
            ds, batch_size=batch_size, shuffle=shuffle
        )
        return ds, dataloader

    def load_network(self):
        net = LSTMTextClassifyNetwork(
            vocab_size=self.tokenizer.vocab_size,
            num_classes=self.tokenizer.num_classes,
            hidden_size=self.config.hidden_size
        )
        return net

    # noinspection PyTypeChecker
    def load_summary_writer(self):
        writer = None
        if self.summary_dir is not None:
            os.makedirs(self.summary_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=self.summary_dir)
            # 将net对应的执行图添加到summary的可视化中
            writer.add_graph(self.net, (torch.randint(0, 10, (2, 8)), torch.ones((2, 8))))
        return writer

    def train_epoch(self, epoch):
        self.net.train()

        train_bar = tqdm(enumerate(self.train_dataloader))
        for batch_idx, batch in train_bar:
            # 获取当前批次的数据
            token_ids = batch['token_ids']
            token_masks = batch['token_masks']
            label_ids = batch['label_id']

            # 前向过程
            score = self.net(token_ids, token_masks)  # [bs,num_classes]
            loss = self.loss_fn(score, label_ids)

            # 反向过程
            self.opt.zero_grad()  # 重置当前优化器对应的所有参数的梯度为0
            loss.backward()  # 计算和当前损失相同的所有参数的梯度值
            self.opt.step()  # 参数更新

            # 效果评估
            pred_idx = torch.argmax(score.detach(), dim=1)  # 获取预测的类别id
            acc = metrics.accuracy_score(label_ids.cpu().numpy(), pred_idx.cpu().numpy())

            # print(f"Train Epoch {epoch}/{self.total_epoch} Batch {batch_idx} Loss:{loss.item():.3f}")
            train_bar.set_description(
                f"Train Epoch {epoch}/{self.total_epoch} Batch {batch_idx} Loss:{loss.item():.3f} Accuracy: {acc:.3f}")
            if self.writer is not None:
                self.writer.add_scalar('train_losses', loss.item(), self.train_batch_steps)
                self.writer.add_scalar('train_accuracy', acc, self.train_batch_steps)
            self.train_batch_steps += 1

    @torch.no_grad()
    def eval_epoch(self, epoch):
        self.net.eval()
        test_bar = tqdm(enumerate(self.val_dataloader))
        for batch_idx, batch in test_bar:
            # 获取当前批次的数据x + y
            token_ids = batch['token_ids']
            token_masks = batch['token_masks']
            batch_y_test = batch['label_id']

            # 前向过程
            score = self.net(token_ids, token_masks)  # [bs,num_classes]
            loss = self.loss_fn(score, batch_y_test)

            # 效果评估
            pred_idx = torch.argmax(score, dim=1)  # 获取预测的类别id
            acc = metrics.accuracy_score(batch_y_test.cpu().numpy(), pred_idx.cpu().numpy())

            _msg = f"Test Epoch {epoch}/{self.total_epoch} Batch {batch_idx} Batch-number:{token_ids.shape[0]} Loss:{loss.item():.3f} Accuracy:{acc:.3f}"
            # print(_msg)
            test_bar.set_description(_msg)
            if self.writer is not None:
                self.writer.add_scalar('val_losses', loss.item(), self.test_batch_steps)
                self.writer.add_scalar('val_accuracy', acc, self.test_batch_steps)
            self.test_batch_steps += 1

    def save(self, epoch):
        obj = {
            'net': self.net.state_dict(),  # 模型网络对应的所有参数
            'epoch': epoch,
            'date': datetime.now()
        }
        torch.save(obj, self.last_model_path)

    def training(self):
        for epoch in range(self.total_epoch):
            # 训练
            self.train_epoch(epoch)

            # 评估
            self.eval_epoch(epoch)

            # 模型持久化
            self.save(epoch)
        # 关闭writer
        self.writer.close()

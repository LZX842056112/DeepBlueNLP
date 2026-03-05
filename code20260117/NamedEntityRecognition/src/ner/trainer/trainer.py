# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/24 21:27
Create User : 19410
Desc : 训练的入口类
"""
import logging
import os
from datetime import datetime

import torch
from tqdm import tqdm

from .. import export
from ..config import Config
from ..datas.dataset import NerTokenClassifyDataset, build_dataloader
from ..datas.tokenizer import Tokenizer
from ..early_stop import EarlyStop
from ..loss.custom_loss import build_losses
from ..metrics import build_metric_func
from ..models import build_network
from ..optim import build_optim
from ..utils import load_json


class Trainer(object):
    def __init__(self, config: Config):
        super().__init__()

        self.config = config
        self.summary_dir = self.config.summary_dir
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        print(f"当前运行设备为: {self.device}")
        self.best_acc = -1.0
        self.train_batch_steps = 0
        self.test_batch_steps = 0
        self.start_epoch = 0
        self.total_epoch = self.config.total_epoch
        os.makedirs(self.config.model_output_dir, exist_ok=True)
        self.last_model_path = os.path.join(self.config.model_output_dir, "last.pkl")
        self.best_model_path = os.path.join(self.config.model_output_dir, "best.pkl")

        self.append_special_token = False
        self.max_length = None
        if self.config.network_type in ['bert']:
            self.append_special_token = True
            self.max_length = min(512, self.config.max_length or 512)

        # 0. 提前停止器创建
        self.early_stop = EarlyStop(max_no_improved_epoch=self.config.max_no_improved_epoch)

        # 1. 分词器的构建
        if isinstance(self.config.tokenizer, str):
            self.tokenizer = Tokenizer(
                vocabs=self.config.tokenizer
            )
        else:
            self.tokenizer = self.config.tokenizer
        if (self.config.vocab_size is None) or (self.config.vocab_size != self.tokenizer.vocab_size):
            logging.warning(f"重置配置信息中的词汇表大小值: {self.config.vocab_size} - {self.tokenizer.vocab_size}")
            self.config.vocab_size = self.tokenizer.vocab_size

        # 标签到id的映射mapping
        if isinstance(self.config.label2id, str):
            self.label2id = load_json(
                file=self.config.label2id
            )
        else:
            self.label2id = self.config.label2id
        self.config.num_classes = len(self.label2id)
        self.id2label = {_id: _label for _label, _id in self.label2id.items()}

        # 2. 训练数据构造
        self.train_dataset, self.train_dataloader = self.load_train_dataloader()

        # 3. 验证数据的加载
        self.val_dataset, self.val_dataloader = self.load_eval_dataloader()

        # 4. 网络结构创建
        self.net = self.load_network()
        print(f"网络结构:\n{self.net}")

        # 损失函数创建
        self.loss_fn = build_losses()

        # 优化器创建
        self.opt, self.lr_scheduler = build_optim(
            self.net, lr=self.config.lr,
            lr_update_total_iters=self.total_epoch
        )

        # 评估函数的构建
        self.metric_fn = build_metric_func(
            label_id2names=self.id2label
        )

        # 可视化日志输出对象的构建
        self.writer = self.load_summary_writer()

        # 进行参数恢复操作
        self.resume_params()

        # 相关对象转移到对应设备上
        self.net.to(device=self.device)
        self.loss_fn.to(device=self.device)

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
        ds = NerTokenClassifyDataset(
            in_file=data_file,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            append_special_token=self.append_special_token,
            max_length=self.max_length
        )
        dataloader = build_dataloader(ds, batch_size, shuffle=shuffle)
        return ds, dataloader

    def load_network(self):
        return build_network(self.config)

    def resume_params(self):
        if os.path.exists(self.best_model_path):
            print(f"参数恢复: {self.best_model_path}")
            ckpt = torch.load(self.best_model_path, map_location='cpu')

            # 模型参数恢复
            net = ckpt['net']
            self.net.load_state_dict(net.state_dict())

            # 其它参数恢复
            self.best_acc = ckpt['acc']
            self.start_epoch = ckpt['epoch'] + 1
            self.total_epoch = self.total_epoch + self.start_epoch

    # noinspection PyTypeChecker
    def load_summary_writer(self):
        writer = None
        if self.summary_dir is not None:
            os.makedirs(self.summary_dir, exist_ok=True)

            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(log_dir=self.summary_dir)
                # 将net对应的执行图添加到summary的可视化中
                writer.add_graph(self.net, (torch.randint(0, 10, (2, 8)), torch.ones((2, 8))))
            except Exception as e:
                logging.error("初始化基于tensorboard的可视化操作对象异常", exc_info=e)
                writer = None

        return writer

    def train_epoch(self, epoch):
        self.net.train()

        train_bar = tqdm(enumerate(self.train_dataloader))
        for batch_idx, batch in train_bar:
            # 获取当前批次的数据
            token_ids = batch['token_ids'].to(device=self.device)  # [bs,t]
            token_masks = batch['token_masks'].to(device=self.device)  # [bs,t]
            label_ids = batch['label_ids'].to(device=self.device)  # [bs,t]

            # 前向过程
            score = self.net(token_ids, token_masks)  # [bs,t,num_classes]
            loss, ner_loss, noner_loss = self.loss_fn(score, label_ids)

            # 反向过程
            self.opt.zero_grad()  # 重置当前优化器对应的所有参数的梯度为0
            loss.backward()  # 计算和当前损失相同的所有参数的梯度值
            self.opt.step()  # 参数更新

            # 效果评估
            metric_values = self.metric_fn(
                [score],
                [token_masks],
                [label_ids]
            )

            _msg = (f"Train Epoch {epoch}/{self.total_epoch} Batch {batch_idx} "
                    f"Loss:{loss.item():.3f} "
                    f"Ner Token Loss:{ner_loss.item():.3f} "
                    f"No Ner Token Loss:{noner_loss.item():.3f} ")
            for _metric_key, _metric_value in metric_values.items():
                _msg += f" {_metric_key}={_metric_value:.3f}"
                if self.writer is not None:
                    self.writer.add_scalar(f'train_{_metric_key}', _metric_value, self.train_batch_steps)
            train_bar.set_description(_msg)
            if self.writer is not None:
                self.writer.add_scalar('train_losses', loss.item(), self.train_batch_steps)
                self.writer.add_scalar('train_ner_losses', ner_loss.item(), self.train_batch_steps)
                self.writer.add_scalar('train_noner_losses', noner_loss.item(), self.train_batch_steps)
            self.train_batch_steps += 1

    @torch.no_grad()
    def eval_epoch(self, epoch):
        self.net.eval()
        test_bar = tqdm(enumerate(self.val_dataloader))
        test_all_scores, test_all_target_id, test_all_masks = [], [], []
        for batch_idx, batch in test_bar:
            # 获取当前批次的数据x + y
            token_ids = batch['token_ids'].to(device=self.device)
            token_masks = batch['token_masks'].to(device=self.device)
            batch_y_test = batch['label_ids'].to(device=self.device)

            # 前向过程
            score, _ = self.net(token_ids, token_masks)  # [bs,t,num_classes]
            loss, ner_loss, noner_loss = self.loss_fn(score, batch_y_test)

            # 效果评估
            metric_values = self.metric_fn(
                [score],
                [token_masks],
                [batch_y_test]
            )

            test_all_scores.append(score.cpu())
            test_all_target_id.append(batch_y_test.cpu())
            test_all_masks.append(token_masks.cpu())

            _msg = (f"Test Epoch {epoch}/{self.total_epoch} Batch {batch_idx} "
                    f"Batch-number:{token_ids.shape[0]} "
                    f"Loss:{loss.item():.3f} "
                    f"Ner Token Loss:{ner_loss.item():.3f} "
                    f"No Ner Token Loss:{noner_loss.item():.3f} ")
            for _metric_key, _metric_value in metric_values.items():
                _msg += f" {_metric_key}={_metric_value:.3f}"
                if self.writer is not None:
                    self.writer.add_scalar(f'val_{_metric_key}', _metric_value, self.test_batch_steps)
            test_bar.set_description(_msg)
            if self.writer is not None:
                self.writer.add_scalar('val_losses', loss.item(), self.test_batch_steps)
                self.writer.add_scalar('val_ner_losses', ner_loss.item(), self.test_batch_steps)
                self.writer.add_scalar('val_noner_losses', noner_loss.item(), self.test_batch_steps)
            self.test_batch_steps += 1

        metric_values = self.metric_fn(
            pred_scores_lst=test_all_scores,
            input_masks_lst=test_all_masks,
            label_ids_lst=test_all_target_id
        )
        _msg = f"Test Epoch {epoch}/{self.total_epoch} "
        for _metric_key, _metric_value in metric_values.items():
            _msg += f" {_metric_key}={_metric_value:.3f}"
            if self.writer is not None:
                self.writer.add_scalar(f'val_epoch_{_metric_key}', _metric_value, global_step=epoch)
        print(_msg)
        return metric_values['best']

    def save(self, epoch, acc):
        obj = {
            'net': self.net,
            'epoch': epoch,
            'date': datetime.now(),
            'acc': acc,
            'label2ids': self.label2id,
            'append_special_token': self.append_special_token,
            'max_length': self.max_length,
            'special_tokens': {
                'cls': self.tokenizer.cls_token,
                'sep': self.tokenizer.sep_token,
                'pad': self.tokenizer.pad_token,
                'unk': self.tokenizer.unk_token
            }
        }
        torch.save(obj, self.last_model_path)
        if self.best_acc < acc:
            torch.save(obj, self.best_model_path)
            self.best_acc = acc

    def training(self):
        for epoch in range(self.start_epoch, self.total_epoch):
            # 输出学习率的信息
            if self.writer is not None:
                for group_idx, group in enumerate(self.opt.param_groups):
                    self.writer.add_scalar(f'lr_{group_idx}', group['lr'], global_step=epoch)

            # 训练
            self.train_epoch(epoch)

            # 评估
            eval_acc = self.eval_epoch(epoch)

            # 模型持久化
            self.save(epoch, eval_acc)

            # 更新学习率
            self.lr_scheduler.step()

            # 进行提前停止判别
            self.early_stop.update(epoch_metric=eval_acc)
            if self.early_stop.is_stop():
                print(f"提前停止模型训练 {epoch}")
                break

        # 将训练好的模型转换为静态结构
        export.export_jit(self.config)
        export.export_onnx(self.config)
        # 关闭writer
        if self.writer is not None:
            self.writer.close()

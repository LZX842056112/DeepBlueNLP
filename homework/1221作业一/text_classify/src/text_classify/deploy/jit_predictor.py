# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/26 20:57
Create User : 19410
Desc : xxx
"""
import json
import os.path

import torch

from ..datas.tokenizer import Tokenizer, ProxyBertTokenizer


class Predictor(object):
    # noinspection PyTypeChecker
    def __init__(self, jit_model_path):
        super().__init__()
        # 1. 模型恢复
        extra_files = {
            'label2ids.txt': '',
            'token2ids.txt': '',
            'network_type.txt': ''
        }
        self.jit_model = torch.jit.load(jit_model_path, map_location="cpu", _extra_files=extra_files)
        self.jit_model.eval().cpu()

        label2ids = json.loads(extra_files['label2ids.txt'])
        self.id2labels = {_id: _label for _label, _id in label2ids.items()}

        network_type = extra_files['network_type.txt']
        network_type = str(network_type, encoding="utf-8")
        if network_type == 'bert':
            self.tokenizer = ProxyBertTokenizer(os.path.dirname(jit_model_path), label2ids)
        else:
            token2ids = json.loads(extra_files['token2ids.txt'])
            self.tokenizer = Tokenizer(token2ids, label2ids)

    @torch.no_grad()
    def predict(self, x: str, k: int = 1):
        # 1. 分词
        token_result = self.tokenizer(x)
        # 2. 构造模型输入数据
        token_ids = torch.tensor([token_result.token_ids], dtype=torch.int64)
        token_masks = torch.ones_like(token_ids, dtype=torch.float32)
        # 3. 调用模型
        score = self.jit_model(token_ids, token_masks)

        # 4. 模型结果处理
        k = max(min(k, len(self.id2labels)), 1)
        prob = torch.softmax(score, dim=-1)
        result = torch.topk(prob, k=k)
        topk_probs = result.values.cpu().numpy()[0]
        topk_indices = result.indices.cpu().numpy()[0]
        topk_class_names = [self.id2labels[_id] for _id in topk_indices]
        final_result = []
        for prob, cls_idx, cls_name in zip(topk_probs, topk_indices, topk_class_names):
            final_result.append({
                'cls_idx': cls_idx,
                'cls_name': cls_name,
                'prob': prob,
            })
        return final_result

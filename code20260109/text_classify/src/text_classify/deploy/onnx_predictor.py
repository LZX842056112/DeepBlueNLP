# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/5 23:02
Create User : 19410
Desc : xxx
"""

import json
import os.path

import numpy as np
import onnxruntime  # pip install onnxruntime==1.22.0 or pip install onnxruntime-gpu

from ..datas.tokenizer import Tokenizer, ProxyBertTokenizer


class Predictor(object):
    # noinspection PyTypeChecker
    def __init__(self, onnx_model_path):
        super().__init__()
        # 1. 模型恢复
        # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        providers = ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)
        self.session = session
        meta = session.get_modelmeta().custom_metadata_map  # metadata

        label2ids = json.loads(meta['label2ids.txt'])
        self.id2labels = {_id: _label for _label, _id in label2ids.items()}

        network_type = meta['network_type.txt']
        if not isinstance(network_type, str):
            network_type = str(network_type, encoding="utf-8")
        if network_type == 'bert':
            self.tokenizer = ProxyBertTokenizer(os.path.dirname(onnx_model_path), label2ids)
        else:
            raise ValueError("不支持非bert模型的恢复!")

    def predict(self, x: str, k: int = 1):
        # 1. 分词
        token_result = self.tokenizer(x)
        # 2. 构造模型输入数据
        token_ids = np.asarray([token_result.token_ids], dtype=np.int64)
        token_masks = np.ones_like(token_ids, dtype=np.float32)

        # 3. 调用模型
        scores = self.session.run(
            ['scores'],
            {"token_ids": token_ids, "token_masks": token_masks}
        )
        scores = scores[0][0]  # [12] ndarray

        # 4. 模型结果处理
        k = max(min(k, len(self.id2labels)), 1)
        probs = np.exp(scores) / np.sum(np.exp(scores))
        topk_indices = np.argsort(probs)[-k:][::-1]
        topk_class_names = [self.id2labels[_id] for _id in topk_indices]
        final_result = []
        for cls_idx, cls_name in zip(topk_indices, topk_class_names):
            final_result.append({
                'cls_idx': int(cls_idx),
                'cls_name': cls_name,
                'prob': float(probs[cls_idx].round(3)),
            })
        return final_result

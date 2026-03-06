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

from ..datas.tokenizer import Tokenizer
from ..utils import trans_entity2tuple, extract_entities


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
        self.tokenizer = Tokenizer(
            vocabs=os.path.join(os.path.dirname(onnx_model_path), "vocab.txt")
        )

        self.append_special_token = eval(meta.get('append_special_token', 'True'))
        self.max_length = eval(meta.get('max_length', '512'))
        print("模型恢复完成!")

    def predict(self, x: str):
        # 1. 分词得到结果数据
        batch = self.tokenizer(
            x,
            append_cls=self.append_special_token,
            append_sep=self.append_special_token,
            max_length=self.max_length,
            return_pt=False
        )

        # 2. 合并数据
        token_ids = []
        token_masks = []
        max_length = 0
        for item in batch:
            token_ids.append(item['token_ids'])
            token_masks.append(item['token_masks'])
            max_length = max(max_length, len(item['token_ids']))
        for i in range(len(token_ids)):
            cur_length = len(token_ids[i])
            if cur_length < max_length:
                token_ids[i].extend([self.tokenizer.pad_token_id for _ in range(max_length - cur_length)])
                token_masks[i].extend([0 for _ in range(max_length - cur_length)])

        # 2. 构造模型输入数据
        token_ids = np.asarray(token_ids, dtype=np.int64)
        token_masks = np.asarray(token_masks, dtype=np.float32)

        # 3. 调用模型得到预测概率 [?,?,25]
        probs = self.session.run(
            ['scores'],
            {"token_ids": token_ids, "token_masks": token_masks}
        )[0]

        # 4. 模型结果处理
        pred_class_ids = np.argmax(probs, axis=-1)  # [?, ?]
        pred_class_ids = (pred_class_ids * token_masks.astype(np.int32)
                          + (1 - token_masks.astype(np.int32)) * (- 100))  # 所有填充位置的预测类别重置为-100
        pred_entities = trans_entity2tuple(
            label_ids=pred_class_ids,
            label_id2names=self.id2labels
        )
        token_lengths = np.sum(token_masks, axis=1)
        final_entities = extract_entities(
            x, pred_entities, sub_text_lengths=token_lengths,
            append_special_token=self.append_special_token
        )
        return final_entities

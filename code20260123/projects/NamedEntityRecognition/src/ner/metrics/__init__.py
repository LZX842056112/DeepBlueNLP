# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/1/11 16:32
Create User : 19410
Desc : xxx
"""
from typing import List, Dict

import torch
from torch import Tensor

from .entity_metrics import list_entity_metrics
from .token_metrics import list_token_accuracy


def build_metric_func(
        label_id2names: Dict[int, str],
        label_pad_id=-100
):
    def _metric(
            pred_scores_lst: List[Tensor],
            input_masks_lst: List[Tensor],
            label_ids_lst: List[Tensor]
    ):
        if input_masks_lst is None:
            input_masks_lst = [
                (label_ids != label_pad_id).to(dtype=torch.float32) for label_ids in label_ids_lst
            ]

        # 计算token级别的准确率
        _token_acc = list_token_accuracy(
            pred_scores_lst,
            input_masks_lst,
            label_ids_lst
        )
        # 计算实体级别的评估指标
        _entity_metric_values = list_entity_metrics(
            pred_scores_lst,
            input_masks_lst,
            label_ids_lst,
            label_id2names,
            label_pad_id=label_pad_id
        )
        _metric_values = {
            'token_acc': _token_acc
        }
        _metric_values.update(_entity_metric_values)
        _metric_values['best'] = _metric_values.get('entity_f1')
        return _metric_values

    return _metric

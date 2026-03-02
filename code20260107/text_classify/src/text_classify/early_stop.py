# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/31 21:07
Create User : 19410
Desc : xxx
"""


class EarlyStop:
    def __init__(self, max_no_improved_epoch=10):
        super().__init__()

        self._is_stop = False
        self._best_metric = 0.0
        self._no_improved_epoch = 0
        self._max_no_improved_epoch = max_no_improved_epoch

    def update(self, epoch_metric: float):
        """
        基于当前的模型评估效果，判断是否需要停止(更新参数)
        :param epoch_metric: 当前epoch对应的模型评估指标
        :return:
        """
        if epoch_metric >= self._best_metric:
            self._best_metric = epoch_metric
            self._no_improved_epoch = 0
        else:
            self._no_improved_epoch += 1

        if self._no_improved_epoch > self._max_no_improved_epoch:
            self._is_stop = True

    def is_stop(self):
        """
        判断是否需要停止训练过程
        :return: True表示需要停止训练过程，False表示不需要停止训练过程
        """
        return self._is_stop

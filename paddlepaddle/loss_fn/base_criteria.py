#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import paddle
from paddle import nn, Tensor
import argparse
from typing import Any


class BaseCriteria(nn.Layer):
    def __init__(self, *args, **kwargs):
        super(BaseCriteria, self).__init__()
        self.eps = 1e-7

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        return parser

    def forward(
        self, input_sample: Any, prediction: Any, target: Any, *args, **kwargs
    ) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def _class_weights(target: Tensor, n_classes: int, norm_val: float = 1.1) -> Tensor:
        # 用来计算张量的直方图
        # 元素被分类为 min 和 max 之间相等宽度的单元格。如果 min 和 max 均为零，则使用数据的最小值和最大值。
        # 小于 min 值和大于 max 的元素将被忽略
        class_hist: Tensor = paddle.histogram(
            target.astype(paddle.float32), bins=n_classes, min=0, max=n_classes - 1
        )
        mask_indices = class_hist == 0

        # normalize between 0 and 1 by dividing by the sum
        norm_hist = paddle.div(class_hist, paddle.to_tensor(class_hist.sum()))
        norm_hist = paddle.add(norm_hist, paddle.to_tensor(norm_val))

        # compute class weights..
        # samples with more frequency will have less weight and vice-versa
        class_wts = paddle.div(paddle.ones_like(class_hist), paddle.log(norm_hist))

        # mask the classes which do not have samples in the current batch
        class_wts[mask_indices] = 0.0

        return class_wts

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

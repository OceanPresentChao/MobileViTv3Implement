#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import paddle
from paddle import nn, Tensor
from typing import Tuple, Union, Any


class BaseModule(nn.Layer):
    """Base class for all modules"""

    def __init__(self, *args, **kwargs):
        super(BaseModule, self).__init__()

    def forward(self, x: Any, *args, **kwargs) -> Any:
        raise NotImplementedError

    def profile_module(self, input: Any, *args, **kwargs) -> Tuple[Any, float, float]:
        raise NotImplementedError

    def __repr__(self):
        return "{}".format(self.__class__.__name__)

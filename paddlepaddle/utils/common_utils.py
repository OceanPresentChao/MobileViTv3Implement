#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import random
import paddle
import numpy as np
import os
import paddle.device.cuda as cuda
from typing import Union, Dict, Optional, Tuple
from paddle import Tensor
from sys import platform

from paddlepaddle.utils import logger
from paddlepaddle.utils.ddp_utils import is_master
from paddlepaddle.cvnets.layers import norm_layers_tuple


def check_compatibility():
    ver = paddle.__version__.split(".")
    major_version = int(ver[0])
    minor_version = int(ver[0])

    if major_version < 1 and minor_version < 7:
        logger.error(
            "Min pytorch version required is 1.7.0. Got: {}".format(".".join(ver))
        )


def check_frozen_norm_layer(model: paddle.nn.Layer) -> Tuple[bool, int]:

    if hasattr(model, "module"):
        model = model.module

    count_norm = 0
    frozen_state = False
    for m in model.modules():
        if isinstance(m, norm_layers_tuple):
            frozen_state = m.weight.requires_grad

    return frozen_state, count_norm


def create_directories(dir_path: str, is_master_node: bool) -> None:
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        if is_master_node:
            logger.log("Directory created at: {}".format(dir_path))
    else:
        if is_master_node:
            logger.log("Directory exists at: {}".format(dir_path))


def move_to_device(
    opts,
    x: Union[Dict, Tensor],
    device: Optional[str] = "cpu",
    non_blocking: Optional[bool] = True,
    *args,
    **kwargs
) -> Union[Dict, Tensor]:

    # if getattr(opts, "dataset.decode_data_on_gpu", False):
    #    # data is already on GPU
    #    return x

    if isinstance(x, Dict):
        # return the tensor because if its already on device
        if "on_gpu" in x and x["on_gpu"]:
            return x

        for k, v in x.items():
            if isinstance(v, Dict):
                x[k] = move_to_device(opts=opts, x=v, device=device)
            elif isinstance(v, Tensor):
                x[k] = v.to(device=device, non_blocking=non_blocking)

    elif isinstance(x, Tensor):
        x = x.to(device=device, non_blocking=non_blocking)
    else:
        logger.error(
            "Inputs of type  Tensor or Dict of Tensors are only supported right now"
        )
    return x


def is_coreml_conversion(opts) -> bool:
    coreml_convert = getattr(opts, "common.enable_coreml_compatible_module", False)
    if coreml_convert:
        return True
    return False

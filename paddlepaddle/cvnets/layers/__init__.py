#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import argparse
import os
import importlib
import inspect

from paddle.nn import MaxPool2D, AvgPool2D

from .base_layer_paddle import BaseLayer
from .conv_layer_paddle import (
    ConvLayer,
    NormActLayer,
    TransposeConvLayer,
    ConvLayer3d,
    SeparableConv,
)
from .flatten_paddle import Flatten
from .linear_layer_paddle import LinearLayer, GroupLinear
from .global_pool_paddle import GlobalPool
from .identity_paddle import Identity
from .non_linear_layers_paddle import get_activation_fn
from .normalization_layers_paddle import get_normalization_layer, norm_layers_tuple
# from .pixel_shuffle import PixelShuffle
# from .upsample import UpSample
# from .pooling import MaxPool2d, AvgPool2d
# from .positional_encoding import SinusoidalPositionalEncoding, LearnablePositionEncoding
from .normalization_layers_paddle import AdjustBatchNormMomentum
from .adaptive_pool_paddle import AdaptiveAvgPool2d
# from .flatten import Flatten
from .multi_head_attention_paddle import MultiHeadAttention
from .dropout_paddle import Dropout, Dropout2d
from .pixel_shuffle_paddle import PixelShuffle
from .positional_encoding import LearnablePositionEncoding, SinusoidalPositionalEncoding
from .single_head_attention_paddle import SingleHeadAttention
# from .softmax import Softmax
from .linear_attention_paddle import LinearSelfAttention

__all__ = [
    "ConvLayer",
    "ConvLayer3d",
    "SeparableConv",
    "NormActLayer",
    "TransposeConvLayer",
    "LinearLayer",
    "GroupLinear",
    "GlobalPool",
    "Identity",
    "PixelShuffle",
    "UpSample",
    "MaxPool2D",
    "AvgPool2D",
    "Dropout",
    "Dropout2d",
    "SinusoidalPositionalEncoding",
    "LearnablePositionEncoding",
    "AdjustBatchNormMomentum",
    "Flatten",
    "MultiHeadAttention",
    "SingleHeadAttention",
    "Softmax",
    "LinearSelfAttention",
]

# iterate through all classes and fetch layer specific arguments
from .softmax_paddle import Softmax
from .upsample import UpSample


def layer_specific_args(parser: argparse.ArgumentParser):
    layer_dir = os.path.dirname(__file__)
    parsed_layers = []
    for file in os.listdir(layer_dir):
        path = os.path.join(layer_dir, file)
        if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
        ):
            layer_name = file[: file.find(
                ".py")] if file.endswith(".py") else file
            module = importlib.import_module("cvnets.layers." + layer_name)
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, BaseLayer) and name not in parsed_layers:
                    parser = cls.add_arguments(parser)
                    parsed_layers.append(name)
    return parser


def arguments_nn_layers(parser: argparse.ArgumentParser):
    # Retrieve layer specific arguments
    parser = layer_specific_args(parser)

    # activation and normalization arguments

    from .activation import arguments_activation_fn

    parser = arguments_activation_fn(parser)

    from .normalization import arguments_norm_layers

    parser = arguments_norm_layers(parser)

    return parser

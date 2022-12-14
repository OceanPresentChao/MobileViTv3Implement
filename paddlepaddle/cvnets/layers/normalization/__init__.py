#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#


import os
import importlib
import argparse

from paddle.nn import BatchNorm3D, BatchNorm2D, BatchNorm1D, InstanceNorm1D, InstanceNorm2D

SUPPORTED_NORM_FNS = []


def register_norm_fn(name):
    def register_fn(fn):
        if name in SUPPORTED_NORM_FNS:
            raise ValueError(
                "Cannot register duplicate normalization function ({})".format(
                    name)
            )
        SUPPORTED_NORM_FNS.append(name)
        return fn

    return register_fn


# automatically import different normalization layers
norm_dir = os.path.dirname(__file__)
for file in os.listdir(norm_dir):
    path = os.path.join(norm_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(
            "paddlepaddle.cvnets.layers.normalization." + model_name)


def arguments_norm_layers(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Normalization layers", description="Normalization layers"
    )

    group.add_argument(
        "--model.normalization.name",
        default=None,
        type=str,
        help="Normalization layer. Defaults to None",
    )
    group.add_argument(
        "--model.normalization.groups",
        default=1,
        type=str,
        help="Number of groups in group normalization layer. Defaults to 1.",
    )
    group.add_argument(
        "--model.normalization.momentum",
        default=0.1,
        type=float,
        help="Momentum in normalization layers. Defaults to 0.1",
    )

    # Adjust momentum in batch norm layers
    group.add_argument(
        "--model.normalization.adjust-bn-momentum.enable",
        action="store_true",
        help="Adjust momentum in batch normalization layers",
    )
    group.add_argument(
        "--model.normalization.adjust-bn-momentum.anneal-type",
        default="cosine",
        type=str,
        help="Method for annealing momentum in Batch normalization layer",
    )
    group.add_argument(
        "--model.normalization.adjust-bn-momentum.final-momentum-value",
        default=1e-6,
        type=float,
        help="Min. momentum in batch normalization layer",
    )

    return parser


# import here to avoid circular loop


__all__ = [
    "BatchNorm3D",
    "BatchNorm2D",
    "BatchNorm1D",
    "GroupNorm",
    "InstanceNorm1D",
    "InstanceNorm2D",
    "SyncBatchNorm",
    "LayerNorm",
    "LayerNorm2D",
]

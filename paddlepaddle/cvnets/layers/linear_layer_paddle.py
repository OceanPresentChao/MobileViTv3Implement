import paddle
from paddle import nn, Tensor
from typing import Optional, Tuple
import argparse
from paddle.nn import functional as F
from paddle.nn import initializer

from paddlepaddle.utils import logger
import math

from .base_layer_paddle import BaseLayer


class LinearLayer(nn.Linear):
    """
    Applies a linear transformation to the input data

    Args:
        in_features (int): number of features in the input tensor
        out_features (int): number of features in the output tensor
        bias  (Optional[bool]): use bias or not
        channel_first (Optional[bool]): Channels are first or last dimension. If first, then use Conv2d

    Shape:
        - Input: :math:`(N, *, C_{in})` if not channel_first else :math:`(N, C_{in}, *)` where :math:`*` means any number of dimensions.
        - Output: :math:`(N, *, C_{out})` if not channel_first else :math:`(N, C_{out}, *)`

    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: Optional[bool] = True,
            channel_first: Optional[bool] = False,
            *args,
            **kwargs
    ) -> None:
        super().__init__(
            weight_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(-1 / math.sqrt(4096), 1 / math.sqrt(4096))),
            bias_attr=paddle.ParamAttr(initializer=nn.initializer.Uniform(
                -1 / math.sqrt(4096), 1 / math.sqrt(4096)) if bias else None),
            in_features=in_features,
            out_features=out_features)

        self.out_features = out_features
        self.in_features = in_features
        self.channel_first = channel_first

        self.reset_params()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model.layer.linear-init",
            type=str,
            default="xavier_uniform",
            help="Init type for linear layers",
        )
        parser.add_argument(
            "--model.layer.linear-init-std-dev",
            type=float,
            default=0.01,
            help="Std deviation for Linear layers",
        )
        return parser

    def reset_params(self):
        if hasattr(self, "weight") and self.weight is not None:
            self.weightAttr = paddle.ParamAttr(
                initializer=initializer.XavierUniform())
            self.weight = self.create_parameter(
                attr=self.weightAttr, shape=self.weight.shape)
        if hasattr(self, "bias") and self.bias is not None:
            self.biasAttr = paddle.ParamAttr(
                initializer=initializer.Constant(0.0))
            self.bias = self.create_parameter(
                attr=self.biasAttr, shape=self.bias.shape)

    def forward(self, x: Tensor) -> Tensor:
        if self.channel_first:
            if not self.training:
                logger.error(
                    "Channel-first mode is only supported during inference")
            if x.dim() != 4:
                logger.error("Input should be 4D, i.e., (B, C, H, W) format")
            # only run during conversion
            with paddle.no_grad():
                return F.conv2d(
                    x=x,
                    weight=self.weight.clone()
                        .detach()
                        .reshape(self.out_features, self.in_features, 1, 1),
                    bias=self.bias,
                )
        else:
            x = F.linear(
                x=x, weight=self.weight, bias=self.bias)
        return x

    def __repr__(self):
        repr_str = (
            "{}(in_features={}, out_features={}, bias={}, channel_first={})".format(
                self.__class__.__name__,
                self.in_features,
                self.out_features,
                True if self.bias is not None else False,
                self.channel_first,
            )
        )
        return repr_str

    def profile_module(
            self, input: Tensor, *args, **kwargs
    ) -> Tuple[Tensor, float, float]:
        out_size = list(input.shape)
        out_size[-1] = self.out_features
        params = sum([p.numel() for p in self.parameters()])
        macs = params
        output = paddle.zeros(
            shape=out_size, dtype=input.dtype, device=input.device)
        return output, params, macs


class GroupLinear(BaseLayer):
    """
    Applies a GroupLinear transformation layer, as defined `here <https://arxiv.org/abs/1808.09029>`_,
    `here <https://arxiv.org/abs/1911.12385>`_ and `here <https://arxiv.org/abs/2008.00623>`_

    Args:
        in_features (int): number of features in the input tensor
        out_features (int): number of features in the output tensor
        n_groups (int): number of groups
        bias (Optional[bool]): use bias or not
        feature_shuffle (Optional[bool]): Shuffle features between groups

    Shape:
        - Input: :math:`(N, *, C_{in})`
        - Output: :math:`(N, *, C_{out})`

    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            n_groups: int,
            bias: Optional[bool] = True,
            feature_shuffle: Optional[bool] = False,
            *args,
            **kwargs
    ) -> None:
        if in_features % n_groups != 0:
            logger.error(
                "Input dimensions ({}) must be divisible by n_groups ({})".format(
                    in_features, n_groups
                )
            )
        if out_features % n_groups != 0:
            logger.error(
                "Output dimensions ({}) must be divisible by n_groups ({})".format(
                    out_features, n_groups
                )
            )

        in_groups = in_features // n_groups
        out_groups = out_features // n_groups

        super().__init__()

        self.weight = nn.Parameter(
            paddle.to_tensor(n_groups, in_groups, out_groups))
        if bias:
            self.bias = nn.Parameter(paddle.to_tensor(n_groups, 1, out_groups))
        else:
            self.bias = None

        self.out_features = out_features
        self.in_features = in_features
        self.n_groups = n_groups
        self.feature_shuffle = feature_shuffle

        self.reset_params()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model.layer.group-linear-init",
            type=str,
            default="xavier_uniform",
            help="Init type for group linear layers",
        )
        parser.add_argument(
            "--model.layer.group-linear-init-std-dev",
            type=float,
            default=0.01,
            help="Std deviation for group linear layers",
        )
        return parser

    def reset_params(self):
        if self.weight is not None:
            self.weight.data = paddle.ParamAttr(
                initializer=initializer.XavierUniform())
        if self.bias is not None:
            self.bias.data = paddle.ParamAttr(
                initializer=initializer.Constant(0.0))

    def _forward(self, x: Tensor) -> Tensor:
        bsz = x.shape[0]
        # [B, N] -->  [B, g, N/g]
        x = x.reshape(bsz, self.n_groups, -1)

        # [B, g, N/g] --> [g, B, N/g]
        x = x.transpose(0, 1)
        # [g, B, N/g] x [g, N/g, M/g] --> [g, B, M/g]
        x: paddle.Tensor = paddle.bmm(x, self.weight)

        if self.bias is not None:
            x = paddle.add(x, self.bias)

        if self.feature_shuffle:
            # [g, B, M/g] --> [B, M/g, g]
            x = x.permute(1, 2, 0)
            # [B, M/g, g] --> [B, g, M/g]
            x = x.reshape(bsz, self.n_groups, -1)
        else:
            # [g, B, M/g] --> [B, g, M/g]
            x = x.transpose(0, 1)

        return x.reshape(bsz, -1)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 2:
            x = self._forward(x)
            return x
        else:
            in_dims = x.shape[:-1]
            n_elements = x.numel() // self.in_features
            x = x.reshape(n_elements, -1)
            x = self._forward(x)
            x = x.reshape(*in_dims, -1)
            return x

    def __repr__(self):
        repr_str = "{}(in_features={}, out_features={}, groups={}, bias={}, shuffle={})".format(
            self.__class__.__name__,
            self.in_features,
            self.out_features,
            self.n_groups,
            True if self.bias is not None else False,
            self.feature_shuffle,
        )
        return repr_str

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        params = sum([p.numel() for p in self.parameters()])
        macs = params

        out_size = list(input.shape)
        out_size[-1] = self.out_features

        output = paddle.zeros(
            shape=out_size, dtype=input.dtype, device=input.device)
        return output, params, macs

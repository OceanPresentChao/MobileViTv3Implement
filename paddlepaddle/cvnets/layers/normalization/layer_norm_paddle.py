#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from paddle import nn, Tensor
from typing import Optional, Union, List, Tuple
import paddle

from . import register_norm_fn


@register_norm_fn(name="layer_norm")
class LayerNorm(nn.LayerNorm):
    """
    Applies `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ over a input tensor

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        epsilon (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine (bool): If ``True``, use learnable affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, *)` where :math:`N` is the batch size
        - Output: same shape as the input
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], any],
        epsilon: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        *args,
        **kwargs
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            epsilon=epsilon,
            # elementwise_affine=elementwise_affine,
        )

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0


@register_norm_fn(name="layer_norm_2d")
class LayerNorm2D(nn.GroupNorm):
    """
    Applies `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ over a 4D input tensor

    Args:
        num_features (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        epsilon (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine (bool): If ``True``, use learnable affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`H` is the input height, and :math:`W` is the input width
        - Output: same shape as the input
    """

    def __init__(
        self,
        num_features: int,
        epsilon: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            num_channels=num_features, epsilon=epsilon, num_groups=1
        )
        self.num_channels = num_features
        self.epsilon = epsilon

    def __repr__(self):
        return "{}(num_channels={}, epsilon={})".format(
            self.__class__.__name__, self.num_channels, self.epsilon
        )

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0

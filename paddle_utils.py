from typing import Optional, Tuple
import paddle
from paddle import Tensor
from paddlepaddle.common import (
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_IMAGE_CHANNELS,
)


def image_size_from_opts(opts) -> Tuple[int, int]:
    try:
        sampler_name = getattr(opts, "sampler.name",
                               "variable_batch_sampler").lower()
        if sampler_name.find("var") > -1:
            im_w = getattr(opts, "sampler.vbs.crop_size_width",
                           DEFAULT_IMAGE_WIDTH)
            im_h = getattr(opts, "sampler.vbs.crop_size_height",
                           DEFAULT_IMAGE_HEIGHT)
        elif sampler_name.find("multi") > -1:
            im_w = getattr(opts, "sampler.msc.crop_size_width",
                           DEFAULT_IMAGE_WIDTH)
            im_h = getattr(opts, "sampler.msc.crop_size_height",
                           DEFAULT_IMAGE_HEIGHT)
        else:
            im_w = getattr(opts, "sampler.bs.crop_size_width",
                           DEFAULT_IMAGE_WIDTH)
            im_h = getattr(opts, "sampler.bs.crop_size_height",
                           DEFAULT_IMAGE_HEIGHT)
    except Exception as e:
        im_h = DEFAULT_IMAGE_HEIGHT
        im_w = DEFAULT_IMAGE_WIDTH
    return im_h, im_w


def create_rand_tensor(
    opts, device: Optional[str] = "cpu", batch_size: Optional[int] = 1
) -> Tensor:
    sampler = getattr(opts, "sampler.name", "batch_sampler")
    im_h, im_w = image_size_from_opts(opts=opts)
    # 注意此处的类型要保持一致为float32
    inp_tensor = paddle.randint_like(
        low=0,
        high=255,
        x=paddle.ones(shape=[batch_size, DEFAULT_IMAGE_CHANNELS, im_h, im_w]),
        dtype=paddle.float32
    )
    inp_tensor = paddle.divide(inp_tensor, paddle.to_tensor(255.0))
    return inp_tensor

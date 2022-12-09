from typing import Optional, Tuple
import paddle
from paddle import Tensor
import sys
import os
import numpy as np
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

def copy_data(input_dir, output_dir, total_num=16):
    img_num = 0
    dirs = os.listdir(input_dir)
    for idx, dir_name in enumerate(dirs):
        temp_input_dir = os.path.join(input_dir, dir_name)
        temp_output_dir = os.path.join(output_dir, dir_name)
        os.makedirs(temp_output_dir, exist_ok=True)
        if img_num >= total_num:
            continue
        img_name = os.listdir(temp_input_dir)[0]
        input_image_name = os.path.join(temp_input_dir, img_name)
        output_image_name = os.path.join(temp_output_dir, img_name)
        cmd = "cp {} {}".format(input_image_name, output_image_name)
        os.system(cmd)
        img_num += 1
    return

def gen_fake_data():
    fake_data = np.random.rand(1, 3, 256, 256).astype(np.float32) - 0.5
    fake_label = np.arange(1).astype(np.int64)
    np.save("./data/fake_data.npy", fake_data)
    np.save("./data/fake_label.npy", fake_label)


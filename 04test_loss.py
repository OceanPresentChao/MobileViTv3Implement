# add loss comparing code
import sys
import torch
import paddle
import numpy as np
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper
from paddlepaddle.options.opts import get_training_arguments 
from reference.loss_fn import build_loss_fn as build_loss_fn_ref
from paddlepaddle.loss_fn import build_loss_fn as build_loss_fn_pad

from utilities import loadModels
sys.argv[1:] = ['--common.config-file',
                './config/classification/imagenet/config.yaml']  # simulate commandline
def test_forward():
  opts = get_training_arguments()
  paddle_model,torch_model = loadModels(opts)

  # init loss
  criterion_paddle = build_loss_fn_pad(opts)
  criterion_torch = build_loss_fn_ref(opts)

  # prepare logger & load data
  reprod_logger = ReprodLogger()
  inputs = np.load("./data/fake_data.npy")
  labels = np.load("./data/fake_label.npy")

  # save the paddle output
  paddle_out = paddle_model(paddle.to_tensor(inputs, dtype="float32"))
  loss_paddle = criterion_paddle(paddle_out,
      paddle_out, paddle.to_tensor(
          labels, dtype="int64"))
  reprod_logger.add("loss", loss_paddle.cpu().detach().numpy())
  reprod_logger.save("./result/loss_paddle.npy")

  # save the torch output
  torch_out = torch_model(torch.tensor(inputs, dtype=torch.float32))
  loss_torch = criterion_torch(torch_out,
      torch_out, torch.tensor(
          labels, dtype=torch.int64))
  reprod_logger.add("loss", loss_torch.cpu().detach().numpy())
  reprod_logger.save("./result/loss_torch.npy")

def compareLoss():
  # load data
  diff_helper = ReprodDiffHelper()
  torch_info = diff_helper.load_info("./result/loss_torch.npy")
  paddle_info = diff_helper.load_info("./result/loss_paddle.npy")

  # compare result and produce log
  diff_helper.compare_info(torch_info, paddle_info)
  diff_helper.report(path="./result/log/loss_diff.log")



if __name__ == "__main__":
    test_forward()
    compareLoss()

    
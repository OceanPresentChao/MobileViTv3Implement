import paddle
import paddle.optimizer
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from paddlepaddle.options.opts import get_training_arguments 

from reference.optim import build_optimizer as build_optimizer_ref
from reference.optim.scheduler import build_scheduler as build_scheduler_ref
from reference.loss_fn import build_loss_fn as build_loss_fn_ref

from paddlepaddle.optim import build_optimizer as build_optimizer_pad
from paddlepaddle.optim.scheduler import build_scheduler as build_scheduler_pad
from paddlepaddle.loss_fn import build_loss_fn as build_loss_fn_pad

from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper
import sys
from utilities import loadModels,train_one_epoch_paddle,train_one_epoch_torch

sys.argv[1:] = ['--common.config-file',
                './config/classification/imagenet/config.yaml']  # simulate commandline

def test_backward():
  max_iter = 3

  # set determinnistic flag
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  FLAGS_cudnn_deterministic = True

  opts = get_training_arguments()
  paddle_model,torch_model = loadModels(opts)

  # init loss
  criterion_paddle = build_loss_fn_pad(opts)
  criterion_torch = build_loss_fn_ref(opts)

  # init optimizer
  opt_paddle = build_optimizer_pad(paddle_model, opts)
  lr_scheduler_paddle = build_scheduler_pad(opts)

  opt_torch = build_optimizer_ref(torch_model, opts)
  lr_scheduler_torch = build_scheduler_ref(opts)

  # prepare logger & load data
  reprod_logger = ReprodLogger()
  inputs = np.load("./data/fake_data.npy")
  labels = np.load("./data/fake_label.npy")

  train_one_epoch_paddle(inputs, labels, paddle_model, criterion_paddle,
                        opt_paddle, lr_scheduler_paddle, max_iter,
                        reprod_logger)

  train_one_epoch_torch(inputs, labels, torch_model, criterion_torch,
                        opt_torch, lr_scheduler_torch, max_iter,
                        reprod_logger)


def compareOptim():
  # load data
  diff_helper = ReprodDiffHelper()
  torch_info = diff_helper.load_info("./result/optim_torch.npy")
  paddle_info = diff_helper.load_info("./result/optim_paddle.npy")

  # compare result and produce log
  diff_helper.compare_info(torch_info, paddle_info)
  diff_helper.report(path="./result/log/backward_diff.log",diff_threshold=3e-5)

if __name__ == "__main__":
    test_backward()
    compareOptim()

    
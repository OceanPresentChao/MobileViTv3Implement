import paddle
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler
from paddlepaddle.options.opts import get_training_arguments 
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper
import sys
from utilities import loadModels
sys.argv[1:] = ['--common.config-file',
                './config/classification/imagenet/config.yaml']  # simulate commandline

def test_backward():
  max_iter = 3
  lr = 1e-3
  momentum = 0.9
  lr_gamma = 0.1

  # set determinnistic flag
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  FLAGS_cudnn_deterministic = True

  opts = get_training_arguments()
  paddle_model,torch_model = loadModels(opts)

  # init loss
  criterion_paddle = paddle.nn.CrossEntropyLoss()
  criterion_torch = torch.nn.CrossEntropyLoss()

  # init optimizer
  lr_scheduler_paddle = paddle.optimizer.lr.StepDecay(
      lr, step_size=max_iter // 3, gamma=lr_gamma)
  opt_paddle = paddle.optimizer.Momentum(
      learning_rate=lr,
      momentum=momentum,
      parameters=paddle_model.parameters())

  opt_torch = torch.optim.SGD(torch_model.parameters(),
                              lr=lr,
                              momentum=momentum)
  lr_scheduler_torch = lr_scheduler.StepLR(
      opt_torch, step_size=max_iter // 3, gamma=lr_gamma)

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
  torch_info = diff_helper.load_info("./result/losses_ref.npy")
  paddle_info = diff_helper.load_info("./result/losses_paddle.npy")

  # compare result and produce log
  diff_helper.compare_info(torch_info, paddle_info)
  diff_helper.report(path="./result/log/backward_diff.log")

if __name__ == "__main__":
    test_backward()

    
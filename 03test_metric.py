import torch
import paddle
import numpy as np
import sys
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper
from paddlepaddle.options.opts import get_training_arguments
from utilities import loadModels, build_paddle_data_pipeline, build_torch_data_pipeline, evaluate
from reference.metrics.topk_accuracy import top_k_accuracy as accuracy_torch
from paddlepaddle.metrics.topk_accuracy import top_k_accuracy as accuracy_paddle

sys.argv[1:] = ['--common.config-file',
                './config/classification/imagenet/config.yaml']  # simulate commandline


def test_forward():
    opts = get_training_arguments()
    # load paddle model
    paddle_model, torch_model = loadModels(opts)

    # prepare logger & load data
    reprod_logger = ReprodLogger()
    paddle_dataset, paddle_dataloader = build_paddle_data_pipeline(opts)
    torch_dataset, torch_dataloader = build_torch_data_pipeline(opts)

    for idx, (paddle_batch, torch_batch
              ) in enumerate(zip(paddle_dataloader, torch_dataloader)):
        if idx > 0:
            break
        evaluate(paddle.to_tensor(paddle_batch["image"]), paddle.to_tensor(paddle_batch["label"]), paddle_model,
                 accuracy_paddle, 'paddle', reprod_logger)
        evaluate(torch.as_tensor(torch_batch["image"]), torch.as_tensor(torch_batch["label"]), torch_model,
                 accuracy_torch,
                 'torch', reprod_logger)


def compareEvaluation():
    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./result/metric_torch.npy")
    paddle_info = diff_helper.load_info("./result/metric_paddle.npy")
    print(torch_info, paddle_info)

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="./result/log/metric_diff.log")


if __name__ == "__main__":
    test_forward()
    compareEvaluation()

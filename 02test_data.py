import os
import sys
import torch
import paddle
import numpy as np
from PIL import Image
from reprod_log import ReprodLogger, ReprodDiffHelper
from paddlepaddle.options.opts import get_training_arguments 
from reference.data.datasets import train_val_datasets as create_datasets_ref
from reference.data.data_loaders import create_train_val_loader as create_loader_ref
from paddlepaddle.data.datasets import train_val_datasets as create_datasets_pad
from paddlepaddle.data.data_loaders import create_train_val_loader as create_loader_pad

sys.argv[1:] = ['--common.config-file',
                './config/classification/imagenet/config.yaml']  # simulate commandline

def build_paddle_data_pipeline(opts):
    train_dataset, valid_dataset = create_datasets_pad(opts)
    train_loader, val_loader, train_sampler = create_loader_pad(opts, train_dataset, valid_dataset)
    return train_dataset, train_loader


def build_torch_data_pipeline(opts):
    train_dataset, valid_dataset = create_datasets_ref(opts)
    train_loader, val_loader, train_sampler = create_loader_ref(opts, train_dataset, valid_dataset)
    return train_dataset, train_loader

def test_data_pipeline():
    opts = get_training_arguments()

    paddle_dataset, paddle_dataloader = build_paddle_data_pipeline(opts)
    print("--------------------------------------")
    torch_dataset, torch_dataloader = build_torch_data_pipeline(opts)

    logger_paddle_data = ReprodLogger()
    logger_torch_data = ReprodLogger()

    logger_paddle_data.add("length", np.array(len(paddle_dataset)))
    logger_torch_data.add("length", np.array(len(torch_dataset)))

    cnt = 0
    for (paddle_batch, torch_batch
              ) in enumerate(zip(paddle_dataloader, torch_dataloader)):
        if cnt >= 5:
            break
        cnt += 1
        logger_paddle_data.add(f"dataloader_{cnt}", paddle_batch[0].numpy())
        logger_torch_data.add(f"dataloader_{cnt}",torch_batch[0].detach().cpu().numpy())
        
    logger_paddle_data.save("./result/data_paddle.npy")
    logger_torch_data.save("./result/data_torch.npy")


def compareData():
    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./result/data_torch.npy")
    paddle_info = diff_helper.load_info("./result/data_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(path="./result/log/data_diff.log")

if __name__ == "__main__":
    test_data_pipeline()

    
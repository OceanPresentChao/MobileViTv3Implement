
import paddle
import torch
import numpy as np
from paddlepaddle.options.opts import get_training_arguments 

from paddlepaddle.cvnets.models.classification import build_classification_model as build_model_pad
from reference.cvnets.models.classification import build_classification_model as build_model_ref

from reference.data.datasets import train_val_datasets as create_datasets_ref
from reference.data.data_loaders import create_train_val_loader as create_loader_ref
from paddlepaddle.data.datasets import train_val_datasets as create_datasets_pad
from paddlepaddle.data.data_loaders import create_train_val_loader as create_loader_pad

def loadModels(opts):
    device = 'cpu'
    torch_device = torch.device("cuda:0" if device == "gpu" else "cpu")
    paddle.set_device(device)


    # load torch model
    torch_model = build_model_ref(opts)
    torch_state_dict = torch.load("./data/torch.pt",map_location=torch_device)
    torch_model.load_state_dict(torch_state_dict)
    torch_model.to(torch_device)
    torch_model.eval()

    # load paddle model
    paddle_model = build_model_pad(opts)
    paddle_state_dict = paddle.load("./data/paddle.pdparams")
    paddle_model.set_state_dict(paddle_state_dict)
    paddle_model.eval()

    return paddle_model,torch_model

def build_paddle_data_pipeline(opts):
    train_dataset, valid_dataset = create_datasets_pad(opts)
    train_loader, val_loader, train_sampler = create_loader_pad(opts, train_dataset, valid_dataset)
    return train_dataset, train_loader


def build_torch_data_pipeline(opts):
    train_dataset, valid_dataset = create_datasets_ref(opts)
    train_loader, val_loader, train_sampler = create_loader_ref(opts, train_dataset, valid_dataset)
    return train_dataset, train_loader


def evaluate(image, labels, model, acc, tag, reprod_logger):
    model.eval()
    output = model(image)

    accracy = acc(output, labels, top_k=(1, 5))

    reprod_logger.add("acc_top1", np.array(accracy[0]))
    reprod_logger.add("acc_top5", np.array(accracy[1]))

    reprod_logger.save("./result/metric_{}.npy".format(tag))

def train_one_epoch_paddle(inputs, labels, model, criterion, optimizer,
                           lr_scheduler, max_iter, reprod_logger):
    for idx in range(max_iter):
        image = paddle.to_tensor(inputs, dtype="float32")
        target = paddle.to_tensor(labels, dtype="int64")
        # import pdb; pdb.set_trace()

        output = model(image)
        print("padddle iter output:",output)
        loss = criterion(image,output, target)
        print("padddle loss:",loss)

        reprod_logger.add("loss_{}".format(idx), loss.cpu().detach().numpy())
        # reprod_logger.add("lr_{}".format(idx), np.array(lr_scheduler.get_lr()))

        optimizer.clear_grad()
        optimizer = lr_scheduler.update_lr(
                optimizer=optimizer, epoch=1, curr_iter=idx
            )
        loss.backward()
        optimizer.step()

    reprod_logger.save("./result/optim_paddle.npy")


def train_one_epoch_torch(inputs, labels, model, criterion, optimizer,
                          lr_scheduler, max_iter, reprod_logger):
    for idx in range(max_iter):
        image = torch.tensor(inputs, dtype=torch.float32)
        target = torch.tensor(labels, dtype=torch.int64)

        output = model(image)
        print("torch iter output:",output)
        loss = criterion(image,output, target)
        print("torch loss:",loss)


        reprod_logger.add("loss_{}".format(idx), loss.cpu().detach().numpy())
        # reprod_logger.add("lr_{}".format(idx),
        #                   np.array(lr_scheduler.get_last_lr()))

        optimizer.zero_grad()
        optimizer = lr_scheduler.update_lr(
                optimizer=optimizer, epoch=1, curr_iter=idx
            )
        loss.backward()
        optimizer.step()

    reprod_logger.save("./result/optim_torch.npy")

    
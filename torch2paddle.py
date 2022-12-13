import numpy as np
import torch
import paddle
from collections import OrderedDict


def torch2paddle():
    torch_path = "./data/torch.pt"
    paddle_path = "./data/paddle.pdparams"
    torch_state_dict = torch.load(torch_path, map_location='cpu')
    fc_names = ["classifier"]
    paddle_state_dict = {}
    # print([key for key in torch_state_dict])
    for k in torch_state_dict:
        if "num_batches_tracked" in k:
            continue
        v = torch_state_dict[k].detach().cpu().numpy()
        flag = [i in k for i in fc_names]
        if any(flag) and "weight" in k:  # ignore bias
            new_shape = [1, 0] + list(range(2, v.ndim))
            print(
                f"name: {k}, ori shape: {v.shape}, new shape: {v.transpose(new_shape).shape}")
            v = v.transpose(new_shape)
        k = k.replace("running_var", "_variance")
        k = k.replace("running_mean", "_mean")
        # if k not in model_state_dict:
        print(k)
        paddle_state_dict[k] = v
    paddle.save(paddle_state_dict, paddle_path)


if __name__ == "__main__":
    torch2paddle()

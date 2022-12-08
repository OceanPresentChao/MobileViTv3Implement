
from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper
from paddlepaddle.options.opts import get_training_arguments 

from paddlepaddle.loss_fn import build_loss_fn as build_loss_pad
from paddlepaddle.cvnets.models.classification import build_classification_model as build_model_pad

from reference.loss_fn import build_loss_fn as build_loss_ref
from reference.cvnets.models.classification import build_classification_model as build_model_ref
import sys
import paddle
import torch
import numpy as np
from torchsummary import summary as torch_summary

sys.argv[1:] = ['--common.config-file',
                './paddlepaddle/config/classification/imagenet/config.yaml']  # simulate commandline

def printParameters(model,filename):
    with open(filename, "w") as f:
            for param in model.parameters():
                f.write(str(param.shape)+str(param))
    f.close()

def compareParam():
    

def test_forword():
    device = 'cpu'
    torch_device = torch.device("cuda:0" if device == "gpu" else "cpu")
    paddle_model,torch_model = loadModels()



    # [batch = 1,channel = 3,h,w]
    inputs = np.load("./data/fake_data.npy")

    # printParameters(paddle_model,"./result/param_paddle.txt")
    # printParameters(torch_model,"./result/param_torch.txt")
    print(paddle.summary(paddle_model, input_size=(1, 3, 256, 256)))
    torch_summary(torch_model, input_size=(3, 256, 256))
    
    return
    # save the paddle output
    reprod_logger = ReprodLogger()
    paddle_out = paddle_model(paddle.to_tensor(inputs, dtype="float32"))
    reprod_logger.add("logits", paddle_out.cpu().detach().numpy())
    reprod_logger.save("./result/forward_paddle.npy")

    # save the torch output
    torch_out = torch_model(
        torch.tensor(
            inputs, dtype=torch.float32).to(torch_device))
    reprod_logger.add("logits", torch_out.cpu().detach().numpy())
    reprod_logger.save("./result/forward_torch.npy")


def loadModels():
    device = 'cpu'
    torch_device = torch.device("cuda:0" if device == "gpu" else "cpu")
    paddle.set_device(device)

    opts = get_training_arguments()

    # load torch model
    torch_model = build_model_ref(opts)
    torch_model.eval()
    torch_state_dict = torch.load("./data/torch.pt",map_location=torch_device)
    torch_model.load_state_dict(torch_state_dict)
    torch_model.to(torch_device)

    # load paddle model
    paddle_model = build_model_pad(opts)
    paddle_model.eval()
    paddle_state_dict = paddle.load("./data/paddle.pdparams")
    paddle_model.set_state_dict(paddle_state_dict)

    

    # load_torch_params(paddle_model, torch_state_dict)

    return paddle_model,torch_model


def load_torch_params(paddle_model, torch_patams):
    paddle_params = paddle_model.state_dict()

    fc_names = ['classifier']
    for key,torch_value in torch_patams.items():
        if 'num_batches_tracked' in key:
            continue
        key = key.replace("running_var", "_variance").replace("running_mean", "_mean").replace("module.", "")
        torch_value = torch_value.detach().cpu().numpy()
        if key in paddle_params:
            flag = [i in key for i in fc_names]
            if any(flag) and "weight" in key :  # ignore bias
                new_shape = [1, 0] + list(range(2, torch_value.ndim))
                print(f"name: {key}, ori shape: {torch_value.shape}, new shape: {torch_value.transpose(new_shape).shape}")
                torch_value = torch_value.transpose(new_shape)
            paddle_params[key] = torch_value
        else:
            print(f'{key} not in paddle')
    paddle_model.set_state_dict(paddle_params)

    
def compareOutput():
    # load data
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("./result/forward_torch.npy")
    paddle_info = diff_helper.load_info("./result/forward_paddle.npy")

    # compare result and produce log
    diff_helper.compare_info(torch_info, paddle_info)
    diff_helper.report(
        path="./result/log/forward_diff.log", diff_threshold=1e-5)

if __name__ == "__main__":
    test_forword()
    # compareOutput()


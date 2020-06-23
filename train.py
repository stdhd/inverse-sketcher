from time import time

from tqdm import tqdm
import torch
import torch.nn
import torch.optim
import numpy as np

#import model
import data
import argparse
import math
from torch.utils.data import DataLoader
import os
import torchvision
import yaml
from model import cINN
from datetime import date
import pprint

# Determine, whether cuda will be enabled
# use x.to_device(args.device)
parser = argparse.ArgumentParser(description='PyTorch')
parser.add_argument('--nocuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
device = None
if not args.nocuda and torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA enabled.")
else:
    device = torch.device('cpu')
    print("CUDA disabled.")

# Define dictionary of hyper parameters
list_hyper_params = ["default.yaml"]

def parse_yaml(file_path: str, create_folder: bool = True) -> dict:
    print("Reading paramfile {}".format(file_path))

    with open(file_path) as f:
        param = yaml.load(f, Loader=yaml.FullLoader)

    if param.get("load_model", False):
        param["save_dir"] = param["load_model"]
    elif create_folder:
        # Find save directory which doesn't exist yet
        save_dir = str(param["model_name"]) + "_" + "".join(str(date.today()).split("-")[1:])
        save_dir_mod = save_dir + "_0"
        i = 1
        while os.path.exists(save_dir_mod):
            save_dir_mod = save_dir + "_" + str(i)
            i += 1

        os.mkdir(save_dir_mod)
        param["save_dir"] = save_dir_mod
    else:
        raise RuntimeError("No folders can be created in read-only mode.")

    return param

def create_dataloaders(data_path, batch_size, test_ratio, split=None):
    data_set = data.ImageDataSet(root_dir=data_path)
    if split is None:
        train_split, test_split = torch.utils.data.random_split(data_set, [math.ceil(len(data_set)*(1-test_ratio)), math.floor(len(data_set)*(test_ratio))])

    dataloader_train = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), drop_last=True)
    dataloader_test = DataLoader(test_split, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

    return dataloader_train, dataloader_test

def get_optimizer(param):
    optimizer = getattr(torch.optim, param.get("optimizer"))(model.parameters(), **param.get("optimizer_params"))
    return optimizer

def get_model(param):
    return cINN(**param.get("model_params")).to(device)


def load_state(param):
    model = get_model(param)
    optimizer = get_optimizer(param)
    state_dicts = torch.load(param.get("load_model")+f"/{param.get('model_type')}.tar", map_location=device)
    model.load_state_dict(state_dicts["model_state_dict"])
    optimizer.load_state_dict(state_dicts["optimizer_state_dict"])
    epoch = state_dicts["epoch"]+1
    opt_param = param.get("optimizer_params")
    if not opt_param.get("weight_decay") is None and not opt_param.get("weight_decay") == optimizer.param_groups[0]["weight_decay"]:
        optimizer.param_groups[0]["weight_decay"] = opt_param["weight_decay"]
    if not opt_param.get("lr") is None and not opt_param.get("lr") == optimizer.param_groups[0]["lr"]:
        optimizer.param_groups[0]["lr"] = opt_param["lr"]
    return model, optimizer, epoch

def save_state(param, model_state, optim_state, epoch, running_loss, overwrite_chkpt=True):
    if not overwrite_chkpt:
        path = os.path.join(save_dir, str(model).split("(")[0] + "_" + str(epoch))
    else:
        path = os.path.join(save_dir, str(model).split("(")[0])
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optim_state,
        'loss': running_loss,
    }, f"{path}.tar")

if __name__ == "__main__":
    # Loop over hyper parameter configurations
    pp = pprint.PrettyPrinter(indent=4)
    for param_name in list_hyper_params:
        params = parse_yaml(os.path.join("params", param_name))
        # initialize model with custom parameters
        #m = model.CINN(params)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(cinn.optimizer, milestones=[20, 40], gamma=0.1)
        dataloader_train, dataloader_test = create_dataloaders("dataset/SketchyDatabase/256x256", params["batch_size"], params["test_ratio"])
        if params.get("load_model", False):
            model, optimizer, epoch = load_state(params)
        else:
            model = get_model(params)
            optimizer = get_optimizer(params)
            epoch = 0

        t_start = time()

        for e in range(params["n_epochs"]):
            epoch += 1
            print("Starting training for params:")
            pp.pprint(params)
            for batch, (sketch, real, label) in enumerate(dataloader_train):
                optimizer.zero_grad()
                sketch, real, label = sketch.to(device), real.to(device), label.to(device)
                gauss_output = model(real, sketch)
                loss = torch.mean(gauss_output**2/2) - torch.mean(model.log_jacobian()) / (gauss_output.shape[1]*gauss_output.shape[2] * gauss_output.shape[3])
                loss.backward()
                optimizer.step()
                print("Loss: {}".format(loss))
            save_state(params, model.state_dict(), optimizer.state_dict(), epoch, loss)
        #### End of training round single hyper parameter setting
        filename = str(params)\
            .replace('{', '')\
            .replace('}', '')\
            .replace("'", '')\
            .replace(' ', '')\
            .replace(':', '')\
            .replace(',', '_')

        # Save state to file
        #torch.save(m.state_dict(), 'output/' + filename + '.pt')
        print('%.3i \t%.6f' % (epoch, (time() - t_start) / 60.))

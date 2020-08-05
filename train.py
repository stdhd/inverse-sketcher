from time import time

from tqdm import tqdm
import torch
import torch.nn
import torch.optim
import numpy as np

import data
import argparse
import math
from torch.utils.data import DataLoader, Subset
import os
import yaml
from architecture import get_model_by_params
from datetime import date
import pprint
from torchvision import transforms
from PIL import ImageFile
import matplotlib.pyplot as plt
import socket
from autoencoder import AutoEncoder


def parse_yaml(file_path: str, create_folder: bool = True) -> dict:
    """
    Create dictionary from yaml file
    :param file_path: path of params yaml file
    :param create_folder: create folder for saved_models, in case has not been created yet
    :return: dictionary of model parameters and training progress, which were loaded from saved_models folder
    """
    print("Reading paramfile {}".format(file_path))
    try:
        with open(file_path) as f:
            param = yaml.load(f, Loader=yaml.FullLoader)
    except:
        raise (RuntimeError("Could not load model parameters from " + file_path + "."))

    if create_folder:
        # Find save directory
        if not os.path.exists("saved_models"):
            os.mkdir("saved_models")
        save_dir = os.path.join("saved_models", str(param["model_name"]) + "_" + "".join(str(date.today()).split("-")[1:]))
        save_dir_mod = save_dir + "_0"
        i = 1
        while os.path.exists(save_dir_mod):
            save_dir_mod = save_dir + "_" + str(i)
            i += 1

        os.mkdir(save_dir_mod)
        param["save_dir"] = save_dir_mod

        # Copy yaml to training dir
        with open(os.path.join (save_dir_mod, 'modelcopy.yaml'), 'w') as outfile:
            yaml.dump(param, outfile, default_flow_style=True)

    else:
        raise RuntimeError("No folders can be created in read-only mode.")

    return param


def get_transform():
    return transforms.Resize((64, 64))


def create_dataloaders(data_path, batch_size, test_ratio, split=None, only_classes=None, only_one_sample=False, load_on_request=False, add_data=None, p=1, bw=False, color=False):
    """
    Create data loaders from ImageDataSet according parameters. If split is provided, it is used by the data loader.
    :param data_path: path of root directory of data set, while directories 'photo' and 'sketch' are sub directories
    :param batch_size:
    :param test_ratio: 0.1 means 10% test data
    :param split: optional train/test split
    :param only_classes: optional list of folder names to retrieve training data from
    :param only_one_sample: Load only one sketch and one image
    :param num_workers: number of workers threads for loading sketches and images from drive
    :return: train and test dataloaders and train and test split
    """
    if bw and color:
        raise(RuntimeError("Can't do both black and white and coloring at the same time."))
    if only_one_sample:
        test_ratio = 0.5
        batch_size = 1

    data_set = data.ImageDataSet(root_dir=data_path, transform=get_transform(), only_classes=only_classes, only_one_sample=only_one_sample, load_on_request=load_on_request, bw=bw, color=color)
    if split is None:
        perm = torch.randperm(len(data_set))
        train_split, test_split = perm[:math.ceil(len(data_set) * (1-test_ratio))], perm[math.ceil(len(data_set) * (1-test_ratio)):]
    else:
        train_split, test_split = split[0], split[1]
    if add_data:
        data_set_add = data.ImageDataSet(root_dir=add_data, transform=get_transform(), only_classes=only_classes, only_one_sample=only_one_sample, load_on_request=load_on_request, bw=bw, color=color)
        dataloader_train = data.CompositeDataloader(DataLoader(Subset(data_set, train_split), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True),
                                                    DataLoader(data_set_add, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True),
                                                    p=p,
                                                    anneal_rate=0.99)
    else:
        dataloader_train = DataLoader(Subset(data_set, train_split), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        dataloader_train.p = p
        dataloader_train.anneal_p = lambda *args : None
    dataloader_test = DataLoader(Subset(data_set, test_split), batch_size=batch_size, shuffle=False, num_workers=0)

    return dataloader_train, dataloader_test, train_split, test_split


def get_optimizer(param, trainables):
    optimizer = getattr(torch.optim, param.get("optimizer"))(trainables, **param.get("optimizer_params"))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=0.98
    )
    return scheduler, optimizer


def get_model(param):
    return get_model_by_params(param).to(device)#cINN(**param.get("model_params")).to(device)


def load_state(param):
    """
    Provide model, optimizer, epoch and split obejcts from given parameter dictionary
    :param param: parameter dictionary of params yaml file
    :return: model, optimizer, epoch, data split
    """
    model = get_model(param)
    scheduler, optimizer = get_optimizer(param, model.parameters())
    try:
        print("saved_models/" + param.get("model_name")+f"/{param.get('model_name')}.tar")
        state_dicts = torch.load("saved_models/" + param.get("model_name")+f"/{param.get('model_name')}.tar", map_location=device)

    except:
        raise (RuntimeError("Could not load training state parameters for " + param.get('model_name') + "."))
    model.model.load_state_dict(state_dicts["model_state_dict"])
    optimizer.load_state_dict(state_dicts["optimizer_state_dict"])
    epoch = state_dicts["epoch"] + 1
    opt_param = param.get("optimizer_params")
    if not opt_param.get("weight_decay") is None and not opt_param.get("weight_decay") == optimizer.param_groups[0]["weight_decay"]:
        optimizer.param_groups[0]["weight_decay"] = opt_param["weight_decay"]
    if not opt_param.get("lr") is None and not opt_param.get("lr") == optimizer.param_groups[0]["lr"]:
        optimizer.param_groups[0]["lr"] = opt_param["lr"]
    split = (state_dicts["train_split"], state_dicts["test_split"])
    try:
        scheduler.load_state_dict(state_dicts["scheduler_state_dict"])
    except:
        print("Warning, could not load scheduler state dict, continuing with default values")
    try:
        p = state_dicts["prob_data1"]
    except:
        p=param.get("prob_data1")
    return model, optimizer, epoch, split, scheduler, p


def save_state(param, model_state, optim_state, scheduler_state, epoch, running_loss, split, p, overwrite_chkpt=True):
    """
    Save state of training into yaml file in folder saved_models
    :param param: dictionary of used parameters
    :param overwrite_chkpt: If true, save to file without _epoch extension. Otherwise save to file with _epoch extension.
    :return:
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    if not overwrite_chkpt:
        path = os.path.join(param["save_dir"], param["model_name"] + "_" + str(epoch))
    else:
        path = os.path.join(param["save_dir"], param["model_name"])
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optim_state,
        'loss': running_loss,
        'train_split': split[0],
        'test_split': split[1],
        'model_params': param['model_params'],
        'batch_size': param['batch_size'],
        'test_ratio': param['test_ratio'],
        'only_classes': param.get('only_classes', None),
        'only_one_sample': param.get('only_one_sample', False),
        'scheduler_state_dict': scheduler_state,
        'architecture': param.get("architecture"),
        'data_path': param.get("data_path"),
        'prob_data1': p
    }, f"{path}.tar")

def validate(model, dataloader_test):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch, (sketch, real, label) in enumerate(tqdm(dataloader_test)):
            sketch, real, label = sketch.to(device), real.to(device), label.to(device)
            if params.get("half_precision", False):
                sketch, real = sketch.half(), real.half()
            gauss_output = model(real, sketch)
            loss = torch.mean(gauss_output**2/2) - torch.mean(model.log_jacobian()) / gauss_output.shape[1]
            val_loss += loss/len(dataloader_test)
    model.train()
    return val_loss

def train_ae(encoder_sizes, decoder_sizes, train_loader, num_epochs=1, bw=False):
    AE = AutoEncoder(encoder_sizes, decoder_sizes, bw=bw).to(device)
    optimizer = torch.optim.Adam(AE.parameters(), lr=0.0005, weight_decay=3e-6)
    for epoch in tqdm(range(num_epochs), "Encoder pretraining"):
        epoch_loss = 0
        for batch, (sketch, real, label) in enumerate(tqdm(dataloader_train, "Batch")):
            optimizer.zero_grad()
            sketch, real, label = sketch.to(device), real.to(device), label.to(device)
            recon = AE(sketch)
            loss = torch.mean((sketch - recon) ** 2)
            loss.backward()
            epoch_loss += loss.item()/len(dataloader_train)
            optimizer.step()
        print("AutoEncoder pretraining: Epoch {} Loss: {}".format(epoch, epoch_loss))
    del AE.decoder
    optimizer.zero_grad()
    del optimizer
    return AE.encoder

if __name__ == "__main__":
    # Determine, whether cuda will be enabled
    # use x.to_device(args.device)
    parser = argparse.ArgumentParser(description='PyTorch')
    parser.add_argument('modelnames', nargs='+', help='model names to train')
    parser.add_argument('--nocuda', help='Disable CUDA', action='store_true')
    parser.add_argument('--nopreload', help='Disable pre-loading of image data', action='store_true')
    parser.add_argument('--nocheckpoints', action='store_true',
                        help='Disable storing training state checkpoints. Model will be saved at the end of training.')
    args = parser.parse_args()
    device = None
    if not args.nocuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA enabled.")
    else:
        device = torch.device('cpu')
        print("CUDA disabled.")

    # Define dictionary of hyper parameters
    if not args.modelnames is None:
        list_hyper_params = args.modelnames
    else:
        print("No model name specified in command line arguments. Will use hard-coded model list...")
        list_hyper_params = ["clamp_glow"]

    # Loop over hyper parameter configurations
    pp = pprint.PrettyPrinter(indent=4)
    for param_name in list_hyper_params:
        if not param_name.endswith(".yaml"):
            param_name = "{}.yaml".format(param_name)
        params = parse_yaml(os.path.join("params", param_name))
        if not 'only_one_sample' in params:
            params['only_one_sample'] = False
        if args.nocheckpoints:
            print("No checkpoint mode active. No checkpoint is created after every batch. Model will be saved to {} at the end of trainig.".format(params["save_dir"]))
        if params['only_one_sample']:
            print("!!!!!!!!!!!!!!!! ONLY ONE SAMPLE MODE IS ACITVE !!!!!!!!!!!!!!!!")
        if not params.get("data_path", False):
            params["data_path"] = "dataset/SketchyDatabase/256x256"

        if params.get("load_model", False):
            # Load training progress from existing split
            model, optimizer, epoch, split, scheduler, p = load_state(params)
            dataloader_train, dataloader_test, train_split, test_split = create_dataloaders(params["data_path"],
                                                                                            params["batch_size"],
                                                                                            params["test_ratio"],
                                                                                            split=split,
                                                                                            only_classes=params.get('only_classes', None),
                                                                                            only_one_sample=params.get('only_one_sample', False),
                                                                                            load_on_request=args.nopreload,
                                                                                            add_data=params.get("add_data"),
                                                                                            p=p,
                                                                                            bw=params["model_params"].get("bw"),
                                                                                            color=params["model_params"].get("color"))
        else:
            # Init new training with new split
            model = get_model(params)
            scheduler, optimizer = get_optimizer(params, model.parameters())
            epoch = 0
            dataloader_train, dataloader_test, train_split, test_split = create_dataloaders(params["data_path"],
                                                                                            params["batch_size"],
                                                                                            params["test_ratio"],
                                                                                            only_classes=params.get('only_classes', None),
                                                                                            only_one_sample=params.get('only_one_sample', False),
                                                                                            load_on_request=args.nopreload,
                                                                                            add_data=params.get("add_data"),
                                                                                            p=params.get("prob_data1"),
                                                                                            bw=params["model_params"].get("bw"),
                                                                                            color=params["model_params"].get("color"))
            split = (train_split, test_split)
        if params.get("half_precision", False):
            model.half()
        t_start = time()
        loss_summary = np.zeros(0)
        print(socket.gethostname())
        print("Starting training for params:")
        pp.pprint(params)
        if params["model_params"].get("pretrain_cond", False) and not params.get("load_model", False):
            del model.cond_net
            model.cond_net = train_ae(params["model_params"]["encoder_sizes"], params["model_params"]["decoder_sizes"], dataloader_train, bw = params['model_params'].get("bw"))
        for e in range(params["n_epochs"]):
            epoch += 1
            epoch_loss = 0
            for batch, (sketch, real, label) in enumerate(tqdm(dataloader_train)):
                optimizer.zero_grad()
                sketch, real, label = sketch.to(device), real.to(device), label.to(device)
                if params.get("half_precision", False):
                    sketch, real = sketch.half(), real.half()
                gauss_output = model(real, sketch)
                loss = torch.mean(gauss_output**2/2) - torch.mean(model.log_jacobian()) / gauss_output.shape[1]
                loss.backward()
                epoch_loss += loss.item()/len(dataloader_train)
                loss_summary = np.append(loss_summary, loss.item())
                optimizer.step()
            dataloader_train.anneal_p()
            scheduler.step()
            np.savetxt(os.path.join(params["save_dir"], 'summary_{}_epoch{}'.format(params["model_name"],  str(epoch))), loss_summary, fmt='%1.3f')
            print("Epoch {} / {} Training Loss: {}, Validation Loss: {}".format(e + 1, params["n_epochs"], epoch_loss, validate(model, dataloader_test)))

            if not args.nocheckpoints:
                save_state(params, model.model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), epoch, epoch_loss, split, dataloader_train.p)

        np.savetxt(os.path.join(params["save_dir"], 'summary_{}_epoch{}_FINAL'.format(params["model_name"], str(epoch))),
                   loss_summary, fmt='%1.3f')
        f = plt.figure()
        plt.plot(loss_summary)
        plt.xlabel('Batch')
        plt.ylabel('Batch Loss')
        plt.savefig(os.path.join(params["save_dir"], 'batchloss_{}.pdf'.format(params["model_name"]) ))
        plt.close()

        if args.nocheckpoints:
            save_state(params, model.model.state_dict(), optimizer.state_dict(), scheduler.state_dict(), epoch, epoch_loss, split, dataloader_train.p)
            print("Model is saved to {}".format(params["save_dir"]))

        print('%.3i \t%.6f min' % (epoch, (time() - t_start) / 60.))

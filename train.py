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
list_hyper_params = [
    {"batch_size": 16, "n_epochs": 5, "learning_rate": 0.01, "test_ratio":0.1},
    {"batch_size": 16, "n_epochs": 5, "learning_rate": 0.001, "test_ratio":0.1}
]

def create_dataloaders(data_path, batch_size, test_ratio, split=None):
    data_set = data.ImageDataSet(root_dir=data_path)
    if split is None:
        train_split, test_split = torch.utils.data.random_split(data_set, [math.ceil(len(data_set)*(1-test_ratio)), math.floor(len(data_set)*(test_ratio))])

    dataloader_train = DataLoader(train_split, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), drop_last=True)
    dataloader_test = DataLoader(test_split, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

    return dataloader_train, dataloader_test

if __name__ == "__main__":
    # Loop over hyper parameter configurations
    for params in list_hyper_params:
        # initialize model with custom parameters
        #m = model.CINN(params)
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(cinn.optimizer, milestones=[20, 40], gamma=0.1)
        dataloader_train, dataloader_test = create_dataloaders("dataset/SketchyDatabase/256x256", params["batch_size"], params["test_ratio"])

        t_start = time()

        for epoch in range(params["n_epochs"]):
            print("Start training for " + str(params))
            for batch, (sketch, real, label) in enumerate(dataloader_train):
                print(sketch.shape, real.shape, label)
                exit()
            #scheduler.step()

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

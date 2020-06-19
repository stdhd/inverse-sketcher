from time import time

from tqdm import tqdm
import torch
import torch.nn
import torch.optim
import numpy as np

import model
import data
import argparse

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
    {"batch_size": 100, "n_epochs": 5, "learning_rate": 0.01},
    {"batch_size": 100, "n_epochs": 5, "learning_rate": 0.001}
]

# Loop over hyper parameter configurations
for params in list_hyper_params:
    # initialize model with custom parameters
    #m = model.CINN(params)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(cinn.optimizer, milestones=[20, 40], gamma=0.1)
    t_start = time()

    for epoch in range(params["n_epochs"]):
        print("Start training for " + str(params))
        for batch, (x, l) in enumerate(data.train_loader):
            pass # TODO training
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

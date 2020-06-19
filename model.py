import torch
import torch.nn as nn
import torch.optim

import FrEIA.framework as Ff
import FrEIA.modules as Fm


class CINN(nn.Module):

    def __init__(self, lr):
        super().__init__()

    def forward(self, x, l):
        pass
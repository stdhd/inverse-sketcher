from aiocoupling import AIO_Block
import torch
import torch.nn as nn
import torch.optim

import FrEIA.framework as Ff
import FrEIA.modules as Fm

def get_model_by_name(name):
    return baseline()

def baseline():
    cond = CondNet()

    nodes = [Ff.InputNode(3, 64, 64)]
    # outputs of the cond. net at different resolution levels
    conditions = [Ff.ConditionNode(64, 64, 64),
                  Ff.ConditionNode(128, 32, 32),
                  Ff.ConditionNode(128, 16, 16),
                  Ff.ConditionNode(512)]

    split_nodes = []

    for k in range(2):
        nodes.append(Ff.Node(nodes[-1], AIO_Block,
                             {'subnet_constructor':SubnetConstructorConv,
                              'num_layers': 2,
                              'conv_size': 3,
                              'dropout': 0.0,
                              'permute_soft': True,
                              'hidden_size': 32,
                              'clamp':1.0},
                             conditions=conditions[0],
                             name = F'block_{k}'))

    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance':0.5}))

    for k in range(4):
        nodes.append(Ff.Node(nodes[-1], AIO_Block,
                             {'subnet_constructor':SubnetConstructorConv,
                              'num_layers': 2,
                              'conv_size': 3 if k%2 else 1,
                              'dropout': 0.0,
                              'permute_soft': True,
                              'hidden_size': 64,
                              'clamp':1.0},
                             conditions=conditions[1],
                             name = F'block_{k+2}'))

    #split off 8/12 ch
    nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                         {'split_size_or_sections':[4,8], 'dim':0}))
    split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))

    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance':0.5}))

    for k in range(4):
        nodes.append(Ff.Node(nodes[-1], AIO_Block,
                             {'subnet_constructor':SubnetConstructorConv,
                              'num_layers': 2,
                              'conv_size': 3 if k%2 else 1,
                              'dropout': 0.0,
                              'permute_soft': True,
                              'hidden_size': 128,
                              'clamp':0.6},
                             conditions=conditions[2],
                             name = F'block_{k+6}'))

    #split off 8/16 ch
    nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                         {'split_size_or_sections':[8,8], 'dim':0}))
    split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))
    nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))

    # fully_connected part
    for k in range(4):
        nodes.append(Ff.Node(nodes[-1], AIO_Block,
                             {'subnet_constructor':SubnetConstructorFC,
                              'num_layers': 2,
                              'dropout': 0.0,
                              'permute_soft': True,
                              'hidden_size': 512,
                              'clamp':0.6},
                             conditions=conditions[3],
                             name = F'block_{k+10}'))
    # concat everything
    nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                         Fm.Concat1d, {'dim':0}))
    nodes.append(Ff.OutputNode(nodes[-1]))
    inn = SketchINN(cond)
    inn.build_inn(nodes, split_nodes, conditions)
    return inn

class CondNet(nn.Module):
    '''conditioning network'''
    def __init__(self):
        super().__init__()

        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(1,  64, 3, padding=1),
                                         nn.LeakyReLU(),
                                         nn.Conv2d(64, 64, 3, padding=1)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.Conv2d(64,  128, 3, padding=1),
                                         nn.LeakyReLU(),
                                         nn.Conv2d(128, 128, 3, padding=1, stride=2)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.Conv2d(128, 128, 3, padding=1, stride=2)),

                           nn.Sequential(nn.LeakyReLU(),
                                         nn.AvgPool2d(4),
                                         Flatten(),
                                         nn.Linear(2048, 512))])


    def forward(self, c):
        outputs = [c]
        for m in self.blocks:
            outputs.append(m(outputs[-1]))
        return outputs[1:]

class SketchINN(nn.Module):
    '''cINN, including the conditioning network'''
    def __init__(self, cond):
        super().__init__()

        self.cond_net = cond



    def build_inn(self, nodes, split_nodes, conditions):

        self.model =  Ff.ReversibleGraphNet(nodes + split_nodes + conditions, verbose=False)
        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.02 * torch.randn_like(p)

    def forward(self, x, c=None, rev=False, intermediate_outputs=False):
        return self.model(x, c=self.cond_net(c), rev=rev, intermediate_outputs=intermediate_outputs)

    def log_jacobian(self):
        return self.model.log_jacobian(run_forward=False)


class Flatten(nn.Module):
    def __init__(self, *args):
        super().__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

class SubnetConstructorFC(nn.Module):
    """This class constructs a subnet for the inner parts of the GLOWCouplingBlocks
    as well as the condition preprocessor.
    size_in: input size of the subnet
    size: output size of the subnet
    internal_size: hidden size of the subnet. If None, set to 2*size
    dropout: dropout chance of the subnet
    """

    def __init__(self, num_layers, size_in, size_out,  internal_size=None, dropout=0.0):
        super().__init__()
        if internal_size is None:
            internal_size = size_out * 2
        if num_layers < 1:
            raise(ValueError("Subnet size has to be 1 or greater"))
        self.layers = []
        for n in range(num_layers):
            input_dim, output_dim = internal_size, internal_size
            if n == 0:
                input_dim = size_in
            if n == num_layers -1:
                output_dim = size_out
            self.layers.append(nn.Linear(input_dim, output_dim))
            if n < num_layers -1:
                self.layers.append(nn.Dropout(p=dropout))
                self.layers.append(nn.ReLU())
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class SubnetConstructorConv(nn.Module):

    def __init__(self, num_layers, size_in, in_channels, hidden_channels, out_channels, stride=1, conv_size=3, dropout=0.0):
        super().__init__()
        if num_layers < 1:
            raise(ValueError("Subnet size has to be 1 or greater"))
        self.layers = []
        ch1 = in_channels
        pad_size = self.get_pad_size(size_in, size_in, stride, conv_size)
        for n in range(num_layers-1):
            self.layers.append(nn.Conv2d(ch1, out_channels, conv_size, padding = pad_size, stride = stride))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            ch1 = out_channels
        self.layers.append(nn.Conv2d(ch1, out_channels, conv_size, padding = pad_size, stride = stride))
        self.layers = nn.ModuleList(self.layers)

    def get_pad_size(self, size_out, size_in, stride, conv_size):
        #size_out == size_in
        #width == height
        return int((size_in/(stride) + conv_size - size_in)//2)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

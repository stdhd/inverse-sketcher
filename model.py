import torch
import torch.nn as nn
import torch.optim

import FrEIA.framework as Ff
import FrEIA.modules as Fm
from aiocoupling import AIO_Block

class GraphNetProcessCondition(Ff.ReversibleGraphNet):
    """Subclass of ReversibleGraphNet, with added condition preprocessing
    node_list: List of FrEIA Nodes, to construct the ReversibleGraphNet from
    condition_constructor: Subnet constructur, needs the options:
        dim_in: input dimensions
        dim_out: output dimensions
        internal_size: hidden dimensions
        dropout: dropout ratio
    This part of the model does NOT need to be invertible.
    cond_dim: input dimensions for preprocessing network
    cond_dim_out: output dimensions for preprocessing network
    internal_size: hidden size of preprocessing network
    cond_dropout: dropout probability of the conditional preprocessing
    ind_in: index of the input node in the node_list !Deprecated!
    ind_out: index of the output node in the node_list !Deprecated!
    verbose: set the ReversibleGraphNet to verbose"""
    def __init__(self,
                 nodes,
                 condition_constructor,
                 num_cond_layers,
                 dim_cond,
                 dim_cond_out,
                 conv_size_cond=3,
                 cond_dropout=0.0,
                 ind_in=None,
                 ind_out=None,
                 verbose=False):
        super().__init__(nodes, ind_in=ind_in, ind_out=ind_out, verbose=verbose)
        self.condition_preprocessor = condition_constructor(num_cond_layers, dim_cond[1], dim_cond[0], dim_cond_out[0], conv_size=conv_size_cond, dropout=cond_dropout)


    def forward(self, x, c=None, rev=False, intermediate_outputs=False):
        cond = self.condition_preprocessor(c)
        result = super().forward(x, c=cond, rev=rev, intermediate_outputs=intermediate_outputs)
        return result

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

def subnet_conv(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256,   3, padding=1), nn.ReLU(),
                         nn.Conv2d(256,  c_out, 3, padding=1))

class SubnetConstructorConv(nn.Module):

    def __init__(self, num_layers, size_in, in_channels, out_channels, stride=1, conv_size=3, dropout=0.0):
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

class cINN(nn.Module):
    """This class wraps the pytorch model and provides utility functions
    keyword args:
        device: pytorch device for computation (cpu/gpu)
        n_blocks: number of GLOWCouplingBlocks in the model
        clamping: output clamping of the subnets. Needed for training stability
        dim_x: dimensions of forward input
        dim_cond: dimensions of the conditional input
        internal_size: width of the hidden layers of the subnets in each block
        internal_size_cond: width of the hidden layers of the condition preprocessing
        dim_cond_out: output width of the condition preprocessing
        cond_dropout: dropout probability of the condition preprocessing
        layers_per_block: number of layers the subnets will have in each block
        dropout: dropout probability of the subnets in the coupling blocks
        num_cond_layers: number of layers the condition preprocessing subnet will have
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.model            = None
        self.params_trainable = None
        self.optim            = None
        self.weight_scheduler = None
        self.train_loader     = None
        self.eval             = False

        #backend
        self.device       = kwargs.get('device')
        self.verbose      = kwargs.get('verbose') == True

        #conditioning network
        self.dim_cond     = kwargs['dim_cond']
        self.conv_size_cond = kwargs.get('conv_size_cond')
        self.dim_cond_out = kwargs.get('dim_cond_out')
        self.cond_dropout  = kwargs.get('cond_dropout')
        self.num_cond_layers = kwargs.get("num_cond_layers")

        #main network
        self.dim_x        = kwargs['dim_x']
        self.n_blocks     = kwargs['n_blocks']
        self.clamping     = kwargs.get('clamping')
        self.conv_size = kwargs.get('conv_size')
        self.layers_per_block = kwargs.get('layers_per_block') #includes input and output layer, so at least 2
        self.dropout = kwargs.get('dropout')
        self.permute_soft = kwargs.get("permute_soft")
        self.define_model_architecture()


    def define_model_architecture(self):
        """Create a GraphNetProcessCondition model based on the settings, using
        SubnetConstructor as the subnet constructor"""
        if self.n_blocks is None or self.dim_x is None or self.dim_cond is None:
            raise(RuntimeError("Model not initialized correctly. Some parameters are undefined."))
        if self.clamping is None:
            self.clamping = 0.9
        if self.dim_cond_out is None:
            self.dim_cond_out = self.dim_cond
        if self.cond_dropout is None:
            self.cond_dropout = 0.0
        if self.dropout is None:
            self.dropout = 0.0
        if self.layers_per_block is None:
            self.layers_per_block = 3
        input_dim = self.dim_x
        nodes = [Ff.InputNode(*input_dim, name='inp')]

        cond_node = Ff.ConditionNode(*self.dim_cond_out)
        #nodes.append(Fm.Node([nodes[-1].out0], Fm.flattening_layer, {}, name='flatten'))

        for i in range(self.n_blocks):
            nodes.append(
                Ff.Node(
                    [nodes[-1].out0],
                    Fm.permute_layer,
                    {'seed':i},
                    name=F'permute_{i}'
                )
            )
            nodes.append(
                Ff.Node(
                    [nodes[-1].out0],
                    AIO_Block,
                    {
                        'clamp':self.clamping,
                        'subnet_constructor': SubnetConstructorConv,
                        'num_layers': self.layers_per_block,
                        'conv_size': self.conv_size,
                        'dropout': self.dropout,
                        'permute_soft' : self.permute_soft == True
                    },
                    conditions = cond_node,
                    name = F'block_{i}'
                )
            )

        nodes.append(Ff.OutputNode([nodes[-1].out0], name='out'))
        nodes.append(cond_node)
        self.model = GraphNetProcessCondition(nodes,
                                            SubnetConstructorConv,
                                            self.num_cond_layers,
                                            self.dim_cond,
                                            self.dim_cond_out,
                                            conv_size_cond = self.conv_size_cond,
                                            cond_dropout=self.cond_dropout,
                                            verbose=self.verbose)
        if not self.device is None:
            self.model.to(self.device)
        self.params_trainable = list(filter(lambda p: p.requires_grad, self.model.parameters()))

    def forward(self, x, c=None, rev=False, intermediate_outputs=False):
        return self.model(x, c=c, rev=rev, intermediate_outputs=intermediate_outputs)

    def log_jacobian(self):
        return self.model.log_jacobian(run_forward=False)

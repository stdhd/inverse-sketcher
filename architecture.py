import torch
import torch.nn as nn
import torch.optim

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from aiocoupling_fc import AIO_Block as AIOFC
from aiocoupling_conv import AIO_Block as AIOCONV
from autoencoder import AutoEncoder


def get_model_by_params(params):
    if params['architecture'].lower() == "glow":
        model =  baseline_glow(params['model_params'])
    elif params['architecture'].lower() == "aio":
        model = baseline_aio(params['model_params'])
    elif params['architecture'].lower() == "bw":
        model = baseline_bw(params['model_params'])
    elif params['architecture'].lower() == "color":
        model = baseline_color(params['model_params'])
    else:
        raise(ValueError("Model architecture is not defined"))
    if params["model_params"].get("pretrain_cond", False):
        del model.cond_net
        model.cond_net = AutoEncoder(params["model_params"]["encoder_sizes"], params["model_params"]["decoder_sizes"], bw = params['model_params'].get("bw")).encoder
    return model

def random_orthog(n):
    w = torch.randn(n, n)
    w = w + w.T
    w, S, V = torch.svd(w)
    return w


def sub_conv(ch_hidden, kernel, num_hidden_layers=0, num_classes=0, batchnorm=True):
    pad = kernel // 2
    if batchnorm:
        if num_classes < 2:
            bn = lambda ch_out: nn.BatchNorm2d(ch_out)
        else:
            bn = lambda ch_out: ConditionalBatchNorm2d(ch_out, num_classes)
    else:
        bn = lambda ch_out: Identity()
    return lambda ch_in, ch_out: nn.Sequential(
        nn.Conv2d(ch_in, ch_hidden, kernel, padding=pad),
        nn.LeakyReLU(),
        *(nn.Conv2d(ch_hidden, ch_hidden, kernel, padding=pad),
        nn.LeakyReLU()) * num_hidden_layers,
        nn.Conv2d(ch_hidden, ch_out, kernel, padding=pad),
        bn(ch_out)
        )


def sub_fc(ch_hidden, num_hidden_layers=0, dropout=0.0, num_classes=0, batchnorm=True):
    if batchnorm:
        if num_classes < 2:
            bn = lambda ch_out: nn.BatchNorm1d(ch_out)
        else:
            bn = lambda ch_out: ConditionalBatchNorm1d(ch_out, num_classes)
    else:
        bn = lambda ch_out: Identity()

    return lambda ch_in, ch_out: nn.Sequential(
        nn.Linear(ch_in, ch_hidden),
        nn.LeakyReLU(),
        *(nn.Linear(ch_hidden, ch_hidden),
        nn.LeakyReLU()) * num_hidden_layers,
        nn.Linear(ch_hidden, ch_out),
        nn.Dropout(p=dropout),
        bn(ch_out)
    )


def baseline_aio(m_params):
    cond = CondNet(m_params)

    nodes = [Ff.InputNode(3, 64, 64)]
    # outputs of the cond. net at different resolution levels
    conditions = [Ff.ConditionNode(64, 64, 64),
                  Ff.ConditionNode(128, 32, 32),
                  Ff.ConditionNode(128, 16, 16),
                  Ff.ConditionNode(512)]

    #split_nodes = []

    for k in range(8):
        nodes.append(Ff.Node(nodes[-1], AIOCONV,
                             {'subnet_constructor': sub_conv(64, 3, 2),
                              'clamp': 1.0},
                             conditions=conditions[0],
                             name=F'block_{k}'))
        #nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
        #nodes.append(Ff.Node([nodes[-1].out0], Fm.conv_1x1, {'M':random_orthog(3)}))
        nodes.append(Ff.Node(nodes[-1], LearnedActNorm , {'M': torch.randn(1), "b": torch.randn(1)}))


    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance': 0.5}))

    for k in range(10):
        nodes.append(Ff.Node(nodes[-1], AIOCONV,
                             {
                                 'subnet_constructor': sub_conv(128, 3, 2),
                                 'clamp': 0.9,
                             },
                             conditions=conditions[1],
                             name=F'block_{k + 2}'))
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
        #nodes.append(Ff.Node([nodes[-1].out0], Fm.conv_1x1, {'M':random_orthog(12)}))
       # nodes.append(Ff.Node(nodes[-1], LearnedActNorm , {'M': torch.randn(1), "b": torch.randn(1)}))


    # split off 8/12 ch
    #nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
    #                     {'split_size_or_sections': [4, 8], 'dim': 0}))
    #split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))

    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance': 0.5}))

    for k in range(8):
        nodes.append(Ff.Node(nodes[-1], AIOCONV,
                             {
                                 'subnet_constructor': sub_conv(256, 3 if k%2 else 1, 2),
                                 'clamp': 1.0,
                             },
                             conditions=conditions[2],
                             name=F'block_{k + 6}'))
        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
        #nodes.append(Ff.Node([nodes[-1].out0], Fm.conv_1x1, {'M':random_orthog(16)}))
        #nodes.append(Ff.Node(nodes[-1], LearnedActNorm , {'M': torch.randn(1), "b": torch.randn(1)}))


    # split off 8/16 ch
    #nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
    #                     {'split_size_or_sections': [8, 8], 'dim': 0}))
    #split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))
    nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))

    # fully_connected part
    for k in range(4):
        nodes.append(Ff.Node(nodes[-1], AIOFC,
                             {
                                 'clamp': 0.6,
                                 'subnet_constructor': sub_fc(1024, 2)
                             },
                             conditions=conditions[3],
                             name=F'block_{k + 10}'))

        nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
        #nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
        #nodes.append(Ff.Node(nodes[-1], LearnedActNorm , {'M': torch.randn(1), "b": torch.randn(1)}))
    # concat everything
    #nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
    #                     Fm.Concat1d, {'dim': 0}))
    nodes.append(Ff.OutputNode(nodes[-1]))
    inn = SketchINN(cond)
    inn.build_inn(nodes, split_nodes, conditions)
    return inn

def baseline_color(m_params):
    if not m_params['permute'] in ['soft', 'random', 'none', 'false', None, False]:
        raise (RuntimeError("Erros in model params: No 'permute'' selected or not understood."))
    if not m_params['act_norm'] in ['learnednorm', 'movingavg', 'none', 'false', None, False]:
        raise (RuntimeError("Erros in model params: No 'act_norm' selected or not understood."))

    cond = CondNet(m_params)

    nodes = [Ff.InputNode(2, 64, 64)]
    # outputs of the cond. net at different resolution levels
    conditions = [Ff.ConditionNode(64, 64, 64),
                  Ff.ConditionNode(128, 32, 32),
                  Ff.ConditionNode(128, 16, 16),
                  Ff.ConditionNode(512)]

    split_nodes = []
    for k in range(m_params['blocks_per_group'][0]):
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                             {'subnet_constructor': sub_conv(64, m_params['kernel_size_per_group'][0],
                                                             m_params['hidden_layers_per_group'][0], batchnorm=m_params.get("bn", True)),
                              'clamp': m_params['clamping_per_group'][0]},
                             conditions=conditions[0],
                             name=F'block_{k}'))
        if m_params['permute'] == 'random':
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
        elif m_params['permute'] == 'soft':
            nodes.append(Ff.Node([nodes[-1].out0], Fm.conv_1x1, {'M':random_orthog(2)}))
        if m_params['act_norm'] == 'learnednorm':
            nodes.append(Ff.Node(nodes[-1], LearnedActNorm, {'M': torch.randn(1), "b": torch.randn(1)}))
        elif m_params['act_norm'] == 'movingavg':
            nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}))


    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance': 0.5}))

    for k in range(m_params['blocks_per_group'][1]):
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                              {
                                 'subnet_constructor': sub_conv(128, m_params['kernel_size_per_group'][1],
                                                                m_params['hidden_layers_per_group'][1], batchnorm=m_params.get("bn", True)),
                                 'clamp': m_params['clamping_per_group'][1],
                             },
                             conditions=conditions[1],
                             name=F'block_{k + 2}'))
        if m_params['permute'] == 'random':
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
        elif m_params['permute'] == 'soft':
            nodes.append(Ff.Node([nodes[-1].out0], Fm.conv_1x1, {'M':random_orthog(8)}))
        if m_params['act_norm'] == 'learnednorm':
            nodes.append(Ff.Node(nodes[-1], LearnedActNorm, {'M': torch.randn(1), "b": torch.randn(1)}))
        elif m_params['act_norm'] == 'movingavg':
            nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}))


    # split off 8/12 ch
    if m_params.get("split", True):
        nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                             {'split_size_or_sections': [6, 2], 'dim': 0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))

    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance': 0.5}))

    for k in range(m_params['blocks_per_group'][2]):
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                             {
                                 'subnet_constructor': sub_conv(256, m_params['kernel_size_per_group'][2][k],
                                                                m_params['hidden_layers_per_group'][2], batchnorm=m_params.get("bn", True)),
                                 'clamp': m_params['clamping_per_group'][2],
                             },
                             conditions=conditions[2],
                             name=F'block_{k + 6}'))
        if m_params['permute'] == 'random':
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
        elif m_params['permute'] == 'soft':
            nodes.append(Ff.Node([nodes[-1].out0], Fm.conv_1x1, {'M':random_orthog(24 if m_params.get("split", True) else 32)}))
        if m_params['act_norm'] == 'learnednorm':
            nodes.append(Ff.Node(nodes[-1], LearnedActNorm , {'M': torch.randn(1), "b": torch.randn(1)}))
        elif m_params['act_norm'] == 'movingavg':
            nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}))


    # split off 8/16 ch
    if m_params.get("split", True):
        nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                             {'split_size_or_sections': [12, 12], 'dim': 0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))
    nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))

    # fully_connected part
    for k in range(m_params['blocks_per_group'][3]):
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                             {
                                 'clamp': m_params['clamping_per_group'][3],
                                 'subnet_constructor': sub_fc(m_params['fc_size'], m_params['hidden_layers_per_group'][3], dropout=m_params['dropout_fc'], batchnorm=m_params.get("bn", True))
                             },
                             conditions=conditions[3],
                             name=F'block_{k + 10}'))

        #nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
        #nodes.append(Ff.Node(nodes[-1], LearnedActNorm , {'M': torch.randn(1), "b": torch.randn(1)}))
    # concat everything
    if m_params.get("split", True):
        nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                         Fm.Concat1d, {'dim': 0}))
    nodes.append(Ff.OutputNode(nodes[-1]))
    inn = SketchINN(cond)
    inn.build_inn(nodes, split_nodes, conditions)
    return inn


def baseline_bw(m_params):
    if not m_params['permute'] in ['soft', 'random', 'none', 'false', None, False]:
        raise (RuntimeError("Erros in model params: No 'permute'' selected or not understood."))
    if not m_params['act_norm'] in ['learnednorm', 'movingavg', 'none', 'false', None, False]:
        raise (RuntimeError("Erros in model params: No 'act_norm' selected or not understood."))

    cond = CondNet(m_params)

    nodes = [Ff.InputNode(4, 32, 32)]
    # outputs of the cond. net at different resolution levels
    conditions = [Ff.ConditionNode(64, 32, 32),
                  Ff.ConditionNode(128, 16, 16),
                  Ff.ConditionNode(128, 8, 8),
                  Ff.ConditionNode(512)]

    split_nodes = []
    for k in range(m_params['blocks_per_group'][0]):
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                             {'subnet_constructor': sub_conv(64, m_params['kernel_size_per_group'][0],
                                                             m_params['hidden_layers_per_group'][0], batchnorm=m_params.get("bn", True)),
                              'clamp': m_params['clamping_per_group'][0]},
                             conditions=conditions[0],
                             name=F'block_{k}'))
        if m_params['permute'] == 'random':
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
        elif m_params['permute'] == 'soft':
            nodes.append(Ff.Node([nodes[-1].out0], Fm.conv_1x1, {'M':random_orthog(4)}))
        if m_params['act_norm'] == 'learnednorm':
            nodes.append(Ff.Node(nodes[-1], LearnedActNorm, {'M': torch.randn(1), "b": torch.randn(1)}))
        elif m_params['act_norm'] == 'movingavg':
            nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}))


    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance': 0.5}))

    for k in range(m_params['blocks_per_group'][1]):
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                              {
                                 'subnet_constructor': sub_conv(128, m_params['kernel_size_per_group'][1],
                                                                m_params['hidden_layers_per_group'][1], batchnorm=m_params.get("bn", True)),
                                 'clamp': m_params['clamping_per_group'][1],
                             },
                             conditions=conditions[1],
                             name=F'block_{k + 2}'))
        if m_params['permute'] == 'random':
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
        elif m_params['permute'] == 'soft':
            nodes.append(Ff.Node([nodes[-1].out0], Fm.conv_1x1, {'M':random_orthog(16)}))
        if m_params['act_norm'] == 'learnednorm':
            nodes.append(Ff.Node(nodes[-1], LearnedActNorm, {'M': torch.randn(1), "b": torch.randn(1)}))
        elif m_params['act_norm'] == 'movingavg':
            nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}))


    # split off 8/12 ch
    if m_params.get("split", True):
        nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                             {'split_size_or_sections': [6, 10], 'dim': 0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))

    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance': 0.5}))

    for k in range(m_params['blocks_per_group'][2]):
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                             {
                                 'subnet_constructor': sub_conv(256, m_params['kernel_size_per_group'][2][k],
                                                                m_params['hidden_layers_per_group'][2], batchnorm=m_params.get("bn", True)),
                                 'clamp': m_params['clamping_per_group'][2],
                             },
                             conditions=conditions[2],
                             name=F'block_{k + 6}'))
        if m_params['permute'] == 'random':
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
        elif m_params['permute'] == 'soft':
            nodes.append(Ff.Node([nodes[-1].out0], Fm.conv_1x1, {'M':random_orthog(24 if m_params.get("split", True) else 64)}))
        if m_params['act_norm'] == 'learnednorm':
            nodes.append(Ff.Node(nodes[-1], LearnedActNorm , {'M': torch.randn(1), "b": torch.randn(1)}))
        elif m_params['act_norm'] == 'movingavg':
            nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}))


    # split off 8/16 ch
    if m_params.get("split", True):
        nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                             {'split_size_or_sections': [12, 12], 'dim': 0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))
    nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))

    # fully_connected part
    for k in range(m_params['blocks_per_group'][3]):
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                             {
                                 'clamp': m_params['clamping_per_group'][3],
                                 'subnet_constructor': sub_fc(m_params['fc_size'], m_params['hidden_layers_per_group'][3], dropout=m_params['dropout_fc'], batchnorm=m_params.get("bn", True))
                             },
                             conditions=conditions[3],
                             name=F'block_{k + 10}'))

        #nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
        #nodes.append(Ff.Node(nodes[-1], LearnedActNorm , {'M': torch.randn(1), "b": torch.randn(1)}))
    # concat everything
    if m_params.get("split", True):
        nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                         Fm.Concat1d, {'dim': 0}))
    nodes.append(Ff.OutputNode(nodes[-1]))
    inn = SketchINN(cond)
    inn.build_inn(nodes, split_nodes, conditions)
    return inn

def baseline_glow(m_params):
    if not m_params['permute'] in ['soft', 'random', 'none', 'false', None, False]:
        raise (RuntimeError("Erros in model params: No 'permute'' selected or not understood."))
    if not m_params['act_norm'] in ['learnednorm', 'movingavg', 'none', 'false', None, False]:
        raise (RuntimeError("Erros in model params: No 'act_norm' selected or not understood."))

    cond = CondNet(m_params)

    nodes = [Ff.InputNode(3, 64, 64)]
    # outputs of the cond. net at different resolution levels
    conditions = [Ff.ConditionNode(64, 64, 64),
                  Ff.ConditionNode(128, 32, 32),
                  Ff.ConditionNode(128, 16, 16),
                  Ff.ConditionNode(512)]

    split_nodes = []
    for k in range(m_params['blocks_per_group'][0]):
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                             {'subnet_constructor': sub_conv(64, m_params['kernel_size_per_group'][0],
                                                             m_params['hidden_layers_per_group'][0], batchnorm=m_params.get("bn", True)),
                              'clamp': m_params['clamping_per_group'][0]},
                             conditions=conditions[0],
                             name=F'block_{k}'))
        if m_params['permute'] == 'random':
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
        elif m_params['permute'] == 'soft':
            nodes.append(Ff.Node([nodes[-1].out0], Fm.conv_1x1, {'M':random_orthog(3)}))
        if m_params['act_norm'] == 'learnednorm':
            nodes.append(Ff.Node(nodes[-1], LearnedActNorm, {'M': torch.randn(1), "b": torch.randn(1)}))
        elif m_params['act_norm'] == 'movingavg':
            nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}))


    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance': 0.5}))

    for k in range(m_params['blocks_per_group'][1]):
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                              {
                                 'subnet_constructor': sub_conv(128, m_params['kernel_size_per_group'][1],
                                                                m_params['hidden_layers_per_group'][1], batchnorm=m_params.get("bn", True)),
                                 'clamp': m_params['clamping_per_group'][1],
                             },
                             conditions=conditions[1],
                             name=F'block_{k + 2}'))
        if m_params['permute'] == 'random':
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
        elif m_params['permute'] == 'soft':
            nodes.append(Ff.Node([nodes[-1].out0], Fm.conv_1x1, {'M':random_orthog(12)}))
        if m_params['act_norm'] == 'learnednorm':
            nodes.append(Ff.Node(nodes[-1], LearnedActNorm, {'M': torch.randn(1), "b": torch.randn(1)}))
        elif m_params['act_norm'] == 'movingavg':
            nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}))


    # split off 8/12 ch
    if m_params.get("split", True):
        nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                             {'split_size_or_sections': [4, 8], 'dim': 0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))

    nodes.append(Ff.Node(nodes[-1], Fm.HaarDownsampling, {'rebalance': 0.5}))

    for k in range(m_params['blocks_per_group'][2]):
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                             {
                                 'subnet_constructor': sub_conv(256, m_params['kernel_size_per_group'][2][k],
                                                                m_params['hidden_layers_per_group'][2], batchnorm=m_params.get("bn", True)),
                                 'clamp': m_params['clamping_per_group'][2],
                             },
                             conditions=conditions[2],
                             name=F'block_{k + 6}'))
        if m_params['permute'] == 'random':
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
        elif m_params['permute'] == 'soft':
            nodes.append(Ff.Node([nodes[-1].out0], Fm.conv_1x1, {'M':random_orthog(16 if m_params.get("split", True) else 48)}))
        if m_params['act_norm'] == 'learnednorm':
            nodes.append(Ff.Node(nodes[-1], LearnedActNorm , {'M': torch.randn(1), "b": torch.randn(1)}))
        elif m_params['act_norm'] == 'movingavg':
            nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}))


    # split off 8/16 ch
    if m_params.get("split", True):
        nodes.append(Ff.Node(nodes[-1], Fm.Split1D,
                             {'split_size_or_sections': [8, 8], 'dim': 0}))
        split_nodes.append(Ff.Node(nodes[-1].out1, Fm.Flatten, {}))
    nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))

    # fully_connected part
    for k in range(m_params['blocks_per_group'][3]):
        nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                             {
                                 'clamp': m_params['clamping_per_group'][3],
                                 'subnet_constructor': sub_fc(m_params['fc_size'], m_params['hidden_layers_per_group'][3], dropout=m_params['dropout_fc'], batchnorm=m_params.get("bn", True))
                             },
                             conditions=conditions[3],
                             name=F'block_{k + 10}'))

        #nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
        #nodes.append(Ff.Node(nodes[-1], LearnedActNorm , {'M': torch.randn(1), "b": torch.randn(1)}))
    # concat everything
    if m_params.get("split", True):
        nodes.append(Ff.Node([s.out0 for s in split_nodes] + [nodes[-1].out0],
                         Fm.Concat1d, {'dim': 0}))
    nodes.append(Ff.OutputNode(nodes[-1]))
    inn = SketchINN(cond)
    inn.build_inn(nodes, split_nodes, conditions)
    return inn

class CondNet(nn.Module):
    '''conditioning network'''

    def __init__(self, params):
        super().__init__()
        bw_pool = Identity()
        if params.get("bw"):
            bw_pool = nn.AvgPool2d(2)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(1, 64, 3, padding=1),
                                                   nn.LeakyReLU(),
                                                   nn.Conv2d(64, 64, 3, padding=1),
                                                   bw_pool),

                                     nn.Sequential(nn.LeakyReLU(),
                                                   nn.Conv2d(64, 128, 3, padding=1),
                                                   nn.LeakyReLU(),
                                                   nn.Conv2d(128, 128, 3, padding=1, stride=2)),

                                     nn.Sequential(nn.LeakyReLU(),
                                                   nn.Conv2d(128, 128, 3, padding=1, stride=2)),

                                     nn.Sequential(nn.LeakyReLU(),
                                                   nn.AvgPool2d(4),
                                                   nn.BatchNorm2d(128),
                                                   Flatten(),
                                                   nn.Dropout(params.get("cond_dropout", 0.0)),
                                                   nn.Linear(512, 512))
                                     ]
                                    )

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
        self.model = Ff.ReversibleGraphNet(nodes + split_nodes + conditions, verbose=False)
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


class LearnedActNorm(nn.Module):
    '''Learned transformation according to y = Mx + b, with invertible
    matrix M.'''

    def __init__(self, dims_in, M, b):
        super().__init__()

        self.M = nn.Parameter(M, requires_grad=True)
        self.b = nn.Parameter(b, requires_grad=True)
        #if len(dims_in) > 1:
        #    self.n_pixels = dims_in[1] * dims_in[2]
        #else:
        self.n_pixels = 1
        self.activation = nn.Softplus(beta=0.5)

    def forward(self, x, rev=False):
        if not rev:
            return [self.activation(self.M) * x[0] + self.b]
        else:
            return [(x[0] - self.b)/ self.activation(self.M) ]

    def jacobian(self, x, rev=False):
        return ((-1)**rev * self.n_pixels) * (torch.log(self.activation(self.M) + 1e-12).sum())

    def output_dims(self, input_dims):
        return input_dims

#taken from https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out

class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
        self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features) * out + beta.view(-1, self.num_features)
        return out

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

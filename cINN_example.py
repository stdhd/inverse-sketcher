from FrEIA.framework import *
from FrEIA.modules import *
import loss
import physics
import time
import math
import os
import torch.nn as nn
from aiocoupling import AIO_Block
from autoencoder import AutoEncoder


class GraphNetProcessCondition(ReversibleGraphNet):
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
                node_list,
                condition_constructor,
                use_encoder,
                num_cond_layers,
                cond_dim_in,
                cond_dim_out,
                internal_size=None,
                cond_dropout=0.0,
                ind_in=None,
                ind_out=None,
                verbose=True,
                pfn=False,
                num_jets=2,
                pfn_layers=1,
                add_num_jets=False):
        super().__init__(node_list, ind_in=ind_in, ind_out=ind_out, verbose=verbose)
        self.pfn = pfn
        self.num_jets = num_jets
        self.cond_dim_out = cond_dim_out
        self.add_num_jets = add_num_jets is True
        self.num_leptons = 2
        self.dim_jet = int(cond_dim_in/(num_jets+self.num_leptons))
        if self.pfn:
            if pfn_layers is None:
                raise(ValueError("No Layer number for post sum pfn network given."))
            cond_dim_in = self.dim_jet
            if pfn_layers > 0:
                print(cond_dim_out + self.dim_jet * self.num_leptons + self.add_num_jets)
                self.sum_preprocessor = condition_constructor(  pfn_layers,
                                                                cond_dim_out + self.dim_jet * self.num_leptons + self.add_num_jets,
                                                                cond_dim_out,
                                                                internal_size,
                                                                cond_dropout)
            else:
                def add_lepton(x):
                    y = x[:,self.dim_jet * self.num_leptons:]
                    y[:,:self.dim_jet * self.num_leptons] += x[:,:self.dim_jet * self.num_leptons]
                    return y
                self.sum_preprocessor = add_lepton

        if not use_encoder:
            self.condition_preprocessor = condition_constructor(num_cond_layers,
                                                                cond_dim_in + (self.add_num_jets and not self.pfn),
                                                                cond_dim_out,
                                                                internal_size,
                                                                cond_dropout)
        else:
            #TODO: implement dropout for AutoEncoder
            print(cond_dim_in + (self.add_num_jets and not self.pfn))
            AE = AutoEncoder(cond_dim_in + (self.add_num_jets and not self.pfn), [internal_size]* (num_cond_layers - 1), [internal_size] * (num_cond_layers - 1), cond_dim_out)
            self.condition_preprocessor = AE.encoder
            self.decoder = AE.decoder


    def forward(self, x, c=None, rev=False, intermediate_outputs=False):
        if not self.pfn:
            num_jets = torch.zeros(c.shape[0]).to(c.device)
            if self.add_num_jets:
                for j in range(self.num_jets):
                    jet = c[:,(self.num_leptons+j) * self.dim_jet:(self.num_leptons+j+1) * self.dim_jet]
                    mask = torch.sum(torch.BoolTensor(jet.cpu() == torch.zeros(jet.shape)), dim = 1).to(c.device) < 4
                    num_jets += mask.float()
                c = torch.cat([c, torch.unsqueeze(num_jets, -1)], dim=1)
            cond = self.condition_preprocessor(c)
        else:
            num_jets = torch.zeros(c.shape[0]).to(c.device)
            cond = torch.zeros(c.shape[0], self.cond_dim_out).to(c.device)
            #for i in range(self.num_leptons):
            #    lepton = c[:,i*self.dim_jet:(i+1)*self.dim_jet]
            #    cond += self.condition_preprocessor(lepton)
            for j in range(self.num_jets):
                jet = c[:,(self.num_leptons+j) * self.dim_jet:(self.num_leptons+j+1) * self.dim_jet]
                mask = torch.sum(torch.BoolTensor(jet.cpu() == torch.zeros(jet.shape)), dim = 1).to(c.device) < 4
                cond += self.condition_preprocessor(jet) * torch.unsqueeze(mask, -1).float()
                num_jets += mask.float()
            if self.add_num_jets:
                cond = torch.cat([cond, torch.unsqueeze(num_jets, -1)], dim=1)
            #print(num_jets)
            cond = self.sum_preprocessor(torch.cat([c[:,:self.num_leptons*self.dim_jet], cond], dim=-1))#cond)
        if not rev and hasattr(self, "whitener"):
            x = self.whitener(x, rev)
        result = super().forward(x, c=cond, rev=rev, intermediate_outputs=intermediate_outputs)
        if rev and hasattr(self, "whitener"):
            result = self.whitener(result, rev)
        return result

class SubnetConstructor(nn.Module):
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


class Whitening:
    """This class performs whitening on the input data.
    data: Training data to calculate the whitening matrix from
    mode: Which whitening mode should be used
        None: No Whitening
        "PCA": Principle Component Analysis -> W = S^(-1/2) * U^T
        "ZCA": Zero Component Analysis -> W = U * S^(-1/2) * U^T
    eps: epsilon for numerical stability (so small eigenvalues don't get overexaggerated influence)
    """
    def __init__(self, data, mode="PCA", eps=1e-7):
        print("Mode " + str(mode))
        self.mode = mode
        self.eps = eps
        self.compute_rotation_matrix(data)


    def compute_rotation_matrix(self, X):
        """Compute the Whitening Matrix and its inverse based on the currently
        set mode and the input data"""
        M = X - torch.mean(X, axis = 0)
        sigma = torch.matmul(M.T, M) / (X.shape[0] - 1)
        eigenvalues, eigenvectors = torch.eig(sigma, True)
        eigenvalues = torch.sqrt(torch.sum(eigenvalues**2, axis=1)) + self.eps
        if self.mode == "PCA":
            self.W = torch.matmul(torch.diag(eigenvalues ** (-1/2)), eigenvectors.T)
        elif self.mode == "ZCA":
            self.W = torch.matmul(eigenvectors, torch.matmul(torch.diag(eigenvalues ** (-1/2)), eigenvectors.T))
        elif self.mode is None:
            self.W = torch.eye(X.shape[1]).to(X.device)
        else:
            raise(RuntimeError("Unknown Whitening mode {}".format(self.mode)))
        self.W_inv = self.W.inverse()
        self.W.requires_grad = False
        self.W_inv.requires_grad = False

    def __call__(self, x, rev=False):
        """Apply the whitening transformation. If rev is True,
        the inverse matrix is used to undo the transformation"""
        if not hasattr(self, "W") or self.W is None:
            raise(RuntimeError("Rotation Matrix has not been computed for whitening"))
        if not rev:
            return torch.matmul(self.W, x.T).T
        else:
            return torch.matmul(self.W_inv, x.T).T

class cINN:
    """This class wraps the pytorch model and provides utility functions
    keyword args:
        device: pytorch device for computation (cpu/gpu)
        scale: the scaling that should be applied to the data
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
        whitening_mode: the whitening that is applied to the data. See class Whitening
        for more information

        lr: learning rate. This will be decayed if training reaches a plateau
        betas: momentum and squared momentum for ADAM optimizer
        weight_decay: weight decay for regularization
        batch_size: batch size during training. MMD Loss profits from large batch sizes
        masses: Average masses of each of the input partons (after decay)
        eps: epsilon for regularizing the ADAM optimizer forget rates

        masses: the mean masses of the two Bosons
        filename: Name the model should be saved as
        id: Particles to predict (pre decay)

        lam_mmd1: lambda for the first particle's MMD invariant mass loss (IML)
        kernel_name1: kernel to be used for the first particle's IML
        kernel_type1: type of the first kernel
            summed: sum over a list of sigmas
            adaptive: choose the sigma for each batch according to its standard deviation
            standard: use a single fixed sigma
        kernel_width1: sigma(s) for first kernel. Either a single integer of a list of sigmas for summed kernel
        lam_mmd2: lambda for the second particle's IML
        kernel_name2: kernel to be used for the second particle's IML
        kernel_type2: type of the second kernel
        kernel_width2: sigma(s) for second kernel.
        n_epochs: number of epochs to train for

        test_ratio: ratio of test data in the whole training set
        checkpoint_save_interval: the intervals at which te model will be saved during training
        checkpoint_save_overwrite: whether to overwrite old or create new checkpoints during training
        max_val_batches: the maximum number of batches to validate the model on
    """
    def __init__(self, **kwargs):


        self.model            = None
        self.params_trainable = None
        self.optim            = None
        self.weight_scheduler = None
        self.train_loader     = None
        self.eval             = False
        self.verbose          = False

        #architecture
        self.device       = kwargs['device']
        self.scale        = kwargs['scale']
        self.n_blocks     = kwargs['n_blocks']
        self.clamping     = kwargs['clamping']
        self.dim_x        = kwargs['dim_x']
        self.dim_cond     = kwargs['dim_cond']
        self.internal_size= kwargs.get('internal_size')
        self.internal_size_cond = kwargs.get('internal_size_cond')
        self.dim_cond_out = kwargs.get('dim_cond_out')
        self.cond_dropout  = kwargs.get('cond_dropout')
        self.layers_per_block = kwargs.get('layers_per_block')
        self.dropout = kwargs.get('dropout')
        self.num_cond_layers = kwargs.get("num_cond_layers")
        self.whitening_mode = kwargs.get("whitening")
        self.use_encoder = kwargs.get("use_encoder")
        self.permute_soft = kwargs.get("permute_soft")
        self.encoder_trained = False
        self.pfn = kwargs.get("pfn")
        self.num_jets = kwargs.get("num_jets")
        self.pfn_layers = kwargs.get("pfn_layers")
        self.add_num_jets = kwargs.get("add_num_jets")
        self.num_leptons = kwargs.get("num_leptons")
        if self.num_leptons is None:
            self.num_leptons = 2
        if self.num_jets is None:
            self.num_jets = 2
        self.dim_jet = int((self.dim_cond - 4*self.num_leptons)/self.num_jets)
        if self.pfn is None:
            self.pfn = False



        #optimizer
        self.lr           = kwargs['lr']
        self.betas        = kwargs['betas']
        self.weight_decay = kwargs['weight_decay']
        self.eps          = kwargs.get('eps')

        #metadata
        self.masses       = kwargs['masses']
        self.filename     = kwargs['filename']
        self.id           = kwargs['id']
        self.num_particles= 1
        if self.id == 'B' or self.id == 'ZH' or self.id == 'ZW':
            self.num_particles += 1
        self.dim_part = int(self.dim_x/self.num_particles)
        self.dim_mass = int(len(self.masses)/self.num_particles)

        #loss
        self.lam_mmd1     = kwargs['lam_mmd1']
        self.lam_mmd2     = kwargs['lam_mmd2']
        self.kernel_type1  = kwargs.get('kernel_type1')
        self.kernel_width1 = kwargs.get('kernel_width1')
        self.kernel_name1  = kwargs.get('kernel_name1')
        self.kernel_type2  = kwargs.get('kernel_type2')
        self.kernel_width2 = kwargs.get('kernel_width2')
        self.kernel_name2  = kwargs.get('kernel_name2')
        self.cond_mmd = kwargs.get('cond_mmd')
        self.single_point_every = kwargs.get('single_point_every')
        self.decay_lam = kwargs.get('lam_decay_factor')
        self.decay_every = kwargs.get('lam_decay_every')
        if not ((self.decay_lam is None) == (self.decay_every is None)):
            raise(RuntimeError("Got a value for only one of the lambda mmd decay parameters\n Decay fator: {}, Decay every: {}").format(self.decay_lam, self.decay_every))

        #training
        self.n_encoder_epochs = kwargs.get('n_encoder_epochs')
        self.n_epochs     = kwargs['n_epochs']
        self.batch_size   = kwargs['batch_size']
        self.gaussian_batch_size = kwargs.get('gaussian_batch_size')
        self.test_ratio   = kwargs.get('test_split')
        self.save_path    = os.path.join("models", kwargs.get('filename') + self.id)
        self.save_every   = kwargs.get("checkpoint_save_interval")
        self.save_overwrite = kwargs.get("checkpoint_save_overwrite")
        self.max_val_batches= kwargs.get("max_val_batches")

        if self.gaussian_batch_size is None:
            self.gaussian_batch_size = self.batch_size
        if not (math.ceil(self.batch_size/self.gaussian_batch_size) == math.floor(self.batch_size/self.gaussian_batch_size)):
            print("""Warning, over all batch size is not divisible by batch size
            for gaussian shape loss. Some training data will be left out""")
        self.epoch = 0
        self.define_kernel()


    def define_model_architecture(self):
        """Create a GraphNetProcessCondition model based on the settings, using
        SubnetConstructor as the subnet constructor"""
        if self.n_blocks is None or self.dim_x is None or self.dim_cond is None or self.device is None:
            raise(RuntimeError("Model not initialized correctly. Some parameters are undefined."))
        if self.clamping is None:
            self.clamping = 5.
        if self.dim_cond_out is None:
            self.dim_cond_out = self.dim_cond
        if self.cond_dropout is None:
            self.cond_dropout = 0.0
        if self.dropout is None:
            self.dropout = 0.0
        if self.layers_per_block is None:
            self.layers_per_block = 3
        input_dim = (self.dim_x,1)
        nodes = [InputNode(*input_dim, name='inp')]

        cond_node = ConditionNode(self.dim_cond_out)
        nodes.append(Node([nodes[-1].out0], flattening_layer, {}, name='flatten'))

        for i in range(self.n_blocks):
            nodes.append(
                Node(
                    [nodes[-1].out0],
                    permute_layer,
                    {'seed':i},
                    name=F'permute_{i}'
                )
            )
            nodes.append(
                Node(
                    [nodes[-1].out0],
                    AIO_Block,
                    {
                        'clamp':self.clamping,
                        'subnet_constructor': SubnetConstructor,
                        'internal_size' : self.internal_size,
                        'num_layers' : self.layers_per_block,
                        'dropout' : self.dropout,
                        'permute_soft' : self.permute_soft == True
                    },
                    conditions = cond_node,
                    name = F'block_{i}'
                )
            )

        nodes.append(OutputNode([nodes[-1].out0], name='out'))
        nodes.append(cond_node)
        self.model = GraphNetProcessCondition(nodes,
                                            SubnetConstructor,
                                            self.use_encoder,
                                            self.num_cond_layers,
                                            self.dim_cond,
                                            self.dim_cond_out,
                                            self.internal_size_cond,
                                            self.cond_dropout,
                                            verbose=self.verbose,
                                            pfn=self.pfn,
                                            num_jets=self.num_jets,
                                            pfn_layers = self.pfn_layers,
                                            add_num_jets = self.add_num_jets).to(self.device)
        self.params_trainable = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        if self.use_encoder:
            self.autoencoder_params = list(filter(lambda p: p.requires_grad, self.model.condition_preprocessor.parameters()))
            decoder_params = list(filter(lambda p: p.requires_grad, self.model.decoder.parameters()))
            self.autoencoder_params.extend(decoder_params)

    def set_optimizer(self):
        """Set the optimizer for training. ADAM is used with learning rate decay
        on plateau."""
        if self.eps is None:
            self.eps = 1e-6
        if self.lr is None:
            self.lr = 0.0002
        if self.betas is None:
            self.betas = [0.9, 0.999]
        if self.weight_decay is None:
            self.weight_decay = 1e-5
        else:
            self.eps = float(self.eps)
        self.optim = torch.optim.Adam(
            self.params_trainable,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim,
            factor = 0.4,
            patience=50,
            cooldown=150,
            threshold=5e-5,
            threshold_mode='rel',
            verbose=True
        )
        if self.use_encoder:
            self.optim_encoder = torch.optim.Adam(
                self.autoencoder_params,
                lr=self.lr,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay
            )

    def initialize_data_loaders(self, data_x, data_cond, reset_splits=False):
        """Initialize data loaders with the given input and condition data.
        The test set is picked based on a random split with ratio test_ratio"""
        if not len(data_x) == len(data_cond):
            raise(ValueError("Condition Events Number ({}) and Input Event Number ({}) of the dataset don't match up.".format(len(data_cond), len(data_x))))
        if not data_x.shape[1] == self.dim_x:
            raise(ValueError("Input Event shape {} does not match input dimensions {}.".format(data_x.shape[1], self.dim_x)))
        if not data_cond.shape[1] == self.dim_cond:
            raise(ValueError("Input Event shape {} does not match input dimensions {}.".format(data_cond.shape[1], self.dim_cond)))

        init_split = False
        if not hasattr(self, "test_split") or self.test_split is None:
            init_split = True
        if not hasattr(self, "train_split") or self.train_split is None:
            if init_split == False:
                raise(RuntimeError("Model test_split exists, but train_split is not defined."))
        else:
            if init_split == True:
                raise(RuntimeError("Model train_split exists, but test_split is not defined."))
        if init_split or reset_splits:
            perm = np.random.permutation(len(data_x))
            self.test_split = perm[:math.ceil(len(perm)*self.test_ratio)]
            self.train_split = perm[math.ceil(len(perm)*self.test_ratio):]
        else:
            print("Initialized Data Loaders with predefined splits")
        if len(self.train_split):
            self.initialize_train_loader(data_x[self.train_split], data_cond[self.train_split])
        if len(self.test_split):
            self.initialize_test_loader(data_x[self.test_split], data_cond[self.test_split])

    def initialize_train_loader(self, data_x, data_cond):
        """Set the model's train loader"""
        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_x, data_cond),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=(not self.eval),
            #num_workers=16,
            #pin_memory=True
        )

    def initialize_test_loader(self, data_x, data_cond):
        """Set the model's test loader"""
        self.test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(data_x, data_cond),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=(not self.eval),
            #num_workers=16,
            #pin_memory=True
        )


    def set_whitener(self):
        """Add the whitener to the cINN"""
        if not hasattr(self, "train_loader"):
            raise(RuntimeError("Attempted to set whitener before initializing data loaders"))
        if not hasattr(self, "whitening_mode"):
            self.whitening_mode = None
        self.model.whitener = Whitening(self.train_loader.dataset.tensors[0], self.whitening_mode)
        if self.verbose:
            print("Whitening Matrix: {}\nInverse Whitening Matrix: {}".format(self.model.whitener.W, self.model.whitener.W_inv))

    def define_kernel(self):
        """Handle undefined kernel properties. The default settings are:
            kernel name: cauchy
            kernel type: standard
            kernel width: 1 for standard, [0.1, 1, 10] for summed"""
        if self.kernel_name1 is None:
            self.kernel_name1 = "cauchy"
        if self.kernel_type1 is None:
            self.kernel_type1 = "standard"
        if self.kernel_width1 is None:
            if self.kernel_type1 == "standard":
                self.kernel_width1 = 1.
            elif self.kernel_type1 == "summed":
                self.kernel_width1 = [0.1, 1., 10]
        if self.kernel_name2 is None:
            self.kernel_name2 = "cauchy"
        if self.kernel_type2 is None:
            self.kernel_type2 = "standard"
        if self.kernel_width2 is None:
            if self.kernel_type2 == "standard":
                self.kernel_width2 = 1.
            elif self.kernel_type2 == "summed":
                self.kernel_width2 = [0.1, 1., 10]

    def get_mass_sq(self, samples):
        """Get the squared mass of a batch of partons (after decay).
        If the model only predicts one pair of partons, the second entry of the
        tuple will be empty."""
        if self.scale is None:
            raise(RuntimeError("Tried to calculate mass before setting model scale."))
        return physics.new_torch_inv_mass_squared_2p(
            samples[:,:self.dim_part],
            self.masses[:self.dim_mass] / self.scale
        ), physics.new_torch_inv_mass_squared_2p(
            samples[:,self.dim_part:],
            self.masses[self.dim_mass:] / self.scale
        )

    def validate(self):
        """Validate the model on the test set. Validation will stop if max_val_batches
        is exceeded."""
        loss_tot = 0
        self.model.eval()
        for batch, (x_samps, c_samps) in enumerate(self.test_loader):
            if batch > self.max_val_batches:
                break
            x_samps, c_samps = x_samps.to(self.device), c_samps.to(self.device)
            gauss_output = self.model(x_samps, c_samps)
            loss1 = torch.mean(gauss_output**2/2) - torch.mean(self.model.log_jacobian(run_forward=False)) / gauss_output.shape[1]

            gauss_samples = torch.randn(x_samps.shape).to(self.device)
            invert_samps = self.model(gauss_samples, c_samps, rev=True)
            gen_mass_part_1, gen_mass_part_2 = self.get_mass_sq(invert_samps)
            real_mass_part_1, real_mass_part_2 = self.get_mass_sq(x_samps)

            loss2, loss3, _ = self.compute_mmd(invert_samps, x_samps, c_samps, [[],[]])


            loss_tot += (loss1  + self.lam_mmd1 * loss2 + self.lam_mmd2 * loss3).item()
        return loss_tot

    def train_encoder(self):
        if self.verbose:
            print("Training Encoder")

        losses = []
        for epoch in range(self.n_encoder_epochs):
            if self.verbose:
                print('\r', 'epoch ', epoch, '/', self.n_encoder_epochs)
            epoch_loss = 0
            for batch, (x_samps, c_samps) in enumerate(self.train_loader):
                x_samps, c_samps = x_samps.to(self.device), c_samps.to(self.device)
                self.optim_encoder.zero_grad()
                if self.add_num_jets:
                    num_jets = torch.zeros(c_samps.shape[0]).to(c_samps.device)
                    for j in range(self.num_jets):
                        jet = c_samps[:,(self.num_leptons+j) * self.dim_jet:(self.num_leptons+j+1) * self.dim_jet]
                        mask = torch.sum(torch.BoolTensor(jet.cpu() == torch.zeros(jet.shape)), dim = 1).to(c_samps.device) < 4
                        num_jets += mask.float()
                    c_samps = torch.cat([c_samps, torch.unsqueeze(num_jets, -1)], dim=1)
                recon_c = self.model.decoder(self.model.condition_preprocessor(c_samps, skip_last=False))
                rec_loss = loss.reconstruction_loss(c_samps, recon_c)
                epoch_loss += rec_loss/len(self.train_loader)
                rec_loss.backward()
                self.optim_encoder.step()
            losses.append(epoch_loss)
            if not epoch % 100:
                print(epoch_loss)
        return losses

    def compute_mmd(self, invert_samps, x_samps, c_samps, epoch_sigmas):
        gen_mass_part_1, gen_mass_part_2 = self.get_mass_sq(invert_samps)
        real_mass_part_1, real_mass_part_2 = self.get_mass_sq(x_samps)
        cond_mass_1, cond_mass_2 = self.get_mass_sq(torch.cat((c_samps[:,1:4], c_samps[:,5:8], c_samps[:,9:12], c_samps[:,13:16]), axis=-1))
        loss_fct = loss.mmd

        if not self.cond_mmd is None:
            if self.cond_mmd:
                loss_fct = loss.cond_mmd

        loss2 =  loss_fct(gen_mass_part_1,
                            real_mass_part_1,
                            cond_mass_1,
                            self.kernel_width1,
                            self.kernel_type1,
                            self.kernel_name1)
        #save sigmas if kernel is adaptive
        if self.kernel_type1 == "adaptive":
            sigma = loss2[1]
            loss2 = loss2[0]
            epoch_sigmas[0].append(float(sigma))
            epoch_sigmas[1].append(0)

        loss3 = 0
        if len(real_mass_part_2) > 0:
            loss3 =  loss_fct(gen_mass_part_2,
                                real_mass_part_2,
                                cond_mass_2,
                                self.kernel_width2,
                                self.kernel_type2,
                                self.kernel_name2)
            if self.kernel_type2 == "adaptive":
                sigma = loss3[1]
                loss3 = loss3[0]
                epoch_sigmas[1][-1] = float(sigma)
        return loss2, loss3, epoch_sigmas

    def save_losses(self, epoch_loss, loss1, loss2, loss3, loss_tot, two_part):
        epoch_loss[0] += loss1.item()/len(self.train_loader)
        epoch_loss[1] += loss2.item()/len(self.train_loader)
        if two_part:
            epoch_loss[2] += loss3.item()/len(self.train_loader)
        epoch_loss[3] += loss_tot.item()/len(self.train_loader)
        return epoch_loss

    #@profile
    def train(self):
        """Train the model for n_epochs. During training the loss, learning rates
        and the model will be saved in intervals. If the kernel is adaptive,
        its sigmas will also be saved.
        """
        loss_list = []
        sigma_list = []
        #print(self.optim.param_groups[0]["lr"])
        start_time = time.time()
        self.model.train()
        learning_rates = []
        old_masses = None

        if (not self.encoder_trained) and self.use_encoder:
            self.train_encoder()

        for epoch in range(self.n_epochs):
            if self.verbose:
                print('\r', 'epoch ', epoch, '/', self.n_epochs)
            epoch_loss = np.zeros(4)
            epoch_sigmas = [[],[]]
            for batch, (x_samps, c_samps) in enumerate(self.train_loader):
                print('\r', epoch*len(self.train_loader)+batch, end='')
                # Transfer to GPU
                x_samps, c_samps = x_samps.to(self.device), c_samps.to(self.device)
                self.optim.zero_grad()
                loss1 = 0

                for i in range(math.floor(self.batch_size/self.gaussian_batch_size)):
                    x_small = x_samps[i*self.gaussian_batch_size:(i+1)*self.gaussian_batch_size]
                    c_small = c_samps[i*self.gaussian_batch_size:(i+1)*self.gaussian_batch_size]
                    gauss_output = self.model(x_small, c_small)
                    loss1 += torch.mean(gauss_output**2/2) - torch.mean(self.model.log_jacobian(run_forward=False)) / gauss_output.shape[1]
                loss1 /= math.floor(self.batch_size/self.gaussian_batch_size)
                #print(loss1)
                gauss_samples = torch.randn(x_samps.shape).to(self.device)
                invert_samps = self.model(gauss_samples, c_samps, rev=True)

                #if training reaches nan, quit
                if torch.any(torch.isnan(invert_samps)):
                    raise(RuntimeError("Reached Nan on epoch {}. Quitting".format(epoch)))

                loss2, loss3, epoch_sigmas = self.compute_mmd(invert_samps, x_samps, c_samps, epoch_sigmas)

                loss_tot = loss1 + self.lam_mmd1 *  loss2 + self.lam_mmd2 * loss3


                if not (self.single_point_every is None):
                    if (len(self.train_loader)*epoch + batch % self.single_point_every) == 0:
                        point = torch.floor(torch.rand(1) * self.batch_size).type(torch.long)
                        cond = c_samps[point].repeat(self.batch_size, 1)
                        gauss = torch.randn(x_samps.shape).to(self.device)
                        invert = self.model(gauss, cond, rev=True)
                        loss4, loss5, _ = self.compute_mmd(invert, x_samps, cond, [[],[]])
                        loss_tot += self.lam_mmd1 * loss4 + self.lam_mmd2 * loss5
                #if training diverges, stop
                if not loss_tot < 1e30:
                    print("Warning, Loss of {} exceeds threshold, skipping back propagation\nLoss 1: {}, Loss 2: {}".format(loss_tot.item(), loss1.item(), loss2.item()))
                    return

                #save losses
                epoch_loss = self.save_losses(epoch_loss, loss1, loss2, loss3, loss_tot, self.dim_part < x_samps.shape[1])

                loss_tot.backward()
                self.optim.step()



            if not epoch % 100:
                print("\r", "Invariant Mass Loss:{}\nGaussian Shape Loss:{}\nTotal Loss:{}".format((epoch_loss[1]+epoch_loss[-2])/2, epoch_loss[0], epoch_loss[-1]))
                pass
            self.epoch += 1

            if not self.decay_every is None:
                if self.epoch and (not (self.epoch % self.decay_every)):
                    self.lam_mmd1 *= self.decay_lam
                    if hasattr(self, "lam_mmd2"):
                        self.lam_mmd2 *= self.decay_lam

            #handle learning rates
            self.scheduler.step(self.validate())

            #save the results of this epoch
            learning_rates.append(self.scheduler.optimizer.param_groups[0]['lr'])
            np.save('losses/learning_rates' + self.filename + self.id + '.npy', np.array(learning_rates))
            loss_list.append(epoch_loss)
            if self.kernel_type1 == "adaptive" or self.kernel_type2 == "adaptive":
                sigma_list.append([[np.mean(epoch_sigmas[0]),
                np.std(epoch_sigmas[0])],
                [np.mean(epoch_sigmas[1]),
                np.std(epoch_sigmas[1])]])
                np.save('losses/sigmas' + self.filename + self.id + '.npy', np.array(sigma_list))
            np.save('losses/losses' + self.filename + self.id + '.npy', np.array(loss_list))

            #create a backup of the model if it is time
            if not (epoch % self.save_every):
                if self.save_overwrite:
                    self.save(self.save_path)
                else:
                    self.save(self.save_path + str(epoch))
        print('\n','Training complete')
        print("--- %s seconds ---" % (time.time() - start_time))

        #save the final list of losses, learning rates and sigmas
        np.save('losses/learning_rates' + self.filename + self.id + '.npy', np.array(learning_rates))
        np.save('losses/losses' + self.filename + self.id + '.npy', np.array(loss_list))
        if self.kernel_type1 == "adaptive":
            np.save('losses/sigmas' + self.filename + self.id + '.npy', np.array(sigma_list))

    def eval_mode(self, evaluate = True):
        """Set the model to eval mode if evaluate == True
        or to train mode if evaluate == False. This is needed so the whole
        dataset is used during testing. (As opposed to dropping the last batch)"""
        self.eval = evaluate
        return self.model.eval()


    def save(self, name):
        """Save the model, its optimizer and the test/train split, as well as the epoch"""
        torch.save({'opt':self.optim.state_dict(),
                    'net':self.model.state_dict(),
                    'split': (self.train_split, self.test_split),
                    'epoch': self.epoch}, name)


    def load(self, name):
        """Load the model, its optimizer and the test/train split, as well as the epoch"""
        state_dicts = torch.load(name, map_location=self.device)
        self.model.load_state_dict(state_dicts['net'])
        self.train_split, self.test_split = state_dicts['split']
        try:
            self.epoch = state_dicts["epoch"]
        except:
            self.epoch = 0
            print("Warning: Epoch number not provided in save file, setting to default {}".format(self.epoch))
        try:
            self.optim.load_state_dict(state_dicts['opt'])
        except ValueError:
            print('Cannot load optimizer for some reason or other')
        self.encoder_trained = True
        self.model.to(self.device)

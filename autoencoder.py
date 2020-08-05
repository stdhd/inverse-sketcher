import torch.nn as nn
import torch


class AutoEncoder(nn.Module):

    def __init__(self, encoder_layer_sizes, decoder_layer_sizes, latent_dim=512):

        super(AutoEncoder, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = Encoder(encoder_layer_sizes, latent_dim)
        self.decoder = Decoder(decoder_layer_sizes, latent_dim)

    def forward(self, x):
        z = self.encoder.forward(x)
        recon_x = self.decoder.forward(z[-1])
        return recon_x[-1]


class Encoder(nn.Module):

    def __init__(self, block_sizes, latent_dim):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(1, 64, 3, padding=1),
                                                   *(nn.LeakyReLU(), nn.Conv2d(64, 64, 3, padding=1))*block_sizes[0]),
                                     nn.Sequential(*(nn.LeakyReLU(), nn.Conv2d(64, 64, 3, padding=1))*block_sizes[1],
                                                   nn.LeakyReLU(),
                                                   nn.Conv2d(64, 128, 3, padding=1, stride=2)),
                                     nn.Sequential(*(nn.LeakyReLU(), nn.Conv2d(128, 128, 3, padding=1))*block_sizes[2],
                                                   nn.LeakyReLU(),
                                                   nn.Conv2d(128, 128, 3, padding=1, stride=2)),
                                     nn.Sequential(nn.LeakyReLU(),
                                                   nn.AvgPool2d(4),
                                                   nn.BatchNorm2d(128),
                                                   Flatten(),
                                                   *(nn.Linear(2048, 2048), nn.LeakyReLU())*block_sizes[3],
                                                   nn.Linear(2048, latent_dim))
                                     ]
                                    )


    def forward(self, x):
        outputs = [x]
        for m in self.blocks:
            outputs.append(m(outputs[-1]))
        return outputs[1:]


class Decoder(nn.Module):

    def __init__(self, block_sizes, latent_dim):
        super(Decoder, self).__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(512, 2048),
                                                   *(nn.LeakyReLU(), nn.Linear(2048, 2048))*block_sizes[0],
                                                   nn.LeakyReLU(),
                                                   View((-1, 128, 4, 4)),
                                                   nn.BatchNorm2d(128),
                                                   nn.ConvTranspose2d(128, 128, 6, padding=0, stride=3, output_padding=1),
                                                   ),
                                     nn.Sequential(nn.ConvTranspose2d(128, 128, 3, padding=1, stride=2, output_padding=1),
                                     *(nn.LeakyReLU(), nn.Conv2d(128, 128, 3, padding=1))*block_sizes[1],
                                                   nn.LeakyReLU(),
                                                   ),
                                     nn.Sequential(nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2, output_padding=1),
                                                   *(nn.LeakyReLU(), nn.Conv2d(64, 64, 3, padding=1))*block_sizes[2],
                                                   ),
                                     nn.Sequential(*(nn.LeakyReLU(), nn.Conv2d(64, 64, 3, padding=1))*block_sizes[3],
                                                   nn.Conv2d(64, 1, 3, padding=1))
                                     ]
                                    )

    def forward(self, z):
        outputs = [z]
        for m in self.blocks:
            outputs.append(m(outputs[-1]))
        return outputs[1:]

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)

class Flatten(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class Unpooling(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.maxunp = nn.MaxUnpool2d(size)

    def forward(self, x):
        ind = torch.ones(x.shape).to(x.device, dtype=torch.int64)
        return self.maxunp(x, ind)

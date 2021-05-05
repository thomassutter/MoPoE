import torch
import torch.nn as nn

from utils.utils import Flatten, Unflatten


class EncoderImg(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, flags):
        super(EncoderImg, self).__init__()

        self.flags = flags
        self.shared_encoder = nn.Sequential(                          # input shape (3, 28, 28)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),     # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # -> (128, 4, 4)
            nn.ReLU(),
            Flatten(),                                                # -> (2048)
            nn.Linear(2048, flags.style_dim + flags.class_dim),       # -> (ndim_private + ndim_shared)
            nn.ReLU(),
        )

        # content branch
        self.class_mu = nn.Linear(flags.style_dim + flags.class_dim, flags.class_dim)
        self.class_logvar = nn.Linear(flags.style_dim + flags.class_dim, flags.class_dim)
        # optional style branch
        if flags.factorized_representation:
            self.style_mu = nn.Linear(flags.style_dim + flags.class_dim, flags.style_dim)
            self.style_logvar = nn.Linear(flags.style_dim + flags.class_dim, flags.style_dim)

    def forward(self, x):
        h = self.shared_encoder(x)
        if self.flags.factorized_representation:
            return self.style_mu(h), self.style_logvar(h), self.class_mu(h), \
                   self.class_logvar(h)
        else:
            return None, None, self.class_mu(h), self.class_logvar(h)


class DecoderImg(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, flags):
        super(DecoderImg, self).__init__()
        self.flags = flags
        self.decoder = nn.Sequential(
            nn.Linear(flags.style_dim + flags.class_dim, 2048),                                # -> (2048)
            nn.ReLU(),
            Unflatten((128, 4, 4)),                                                            # -> (128, 4, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),                   # -> (64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (3, 28, 28)
        )

    def forward(self, style_latent_space, class_latent_space):
        if self.flags.factorized_representation:
            z = torch.cat((style_latent_space, class_latent_space), dim=1)
        else:
            z = class_latent_space
        x_hat = self.decoder(z)
        # x_hat = torch.sigmoid(x_hat)
        return x_hat, torch.tensor(0.75).to(z.device)  # NOTE: consider learning scale param, too


# class EncoderImg(nn.Module):
#     def __init__(self, flags):
#         super(EncoderImg, self).__init__()
#         self.flags = flags
#         self.hidden_dim = 400

#         modules = []
#         modules.append(nn.Sequential(nn.Linear(784, self.hidden_dim), nn.ReLU(True)))
#         modules.extend([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))
#                         for _ in range(flags.num_hidden_layers - 1)])
#         self.enc = nn.Sequential(*modules)
#         self.relu = nn.ReLU()
#         if flags.factorized_representation:
#             # style
#             self.style_mu = nn.Linear(in_features=self.hidden_dim, out_features=flags.style_mnist_dim, bias=True)
#             self.style_logvar = nn.Linear(in_features=self.hidden_dim, out_features=flags.style_mnist_dim, bias=True)
#             # class
#             self.class_mu = nn.Linear(in_features=self.hidden_dim, out_features=flags.class_dim, bias=True)
#             self.class_logvar = nn.Linear(in_features=self.hidden_dim, out_features=flags.class_dim, bias=True)
#         else:
#             #non-factorized
#             self.hidden_mu = nn.Linear(in_features=self.hidden_dim, out_features=flags.class_dim, bias=True)
#             self.hidden_logvar = nn.Linear(in_features=self.hidden_dim, out_features=flags.class_dim, bias=True)


#     def forward(self, x):
#         h = x.view(*x.size()[:-3], -1)
#         h = self.enc(h)
#         h = h.view(h.size(0), -1)
#         if self.flags.factorized_representation:
#             style_latent_space_mu = self.style_mu(h)
#             style_latent_space_logvar = self.style_logvar(h)
#             class_latent_space_mu = self.class_mu(h)
#             class_latent_space_logvar = self.class_logvar(h)
#             style_latent_space_mu = style_latent_space_mu.view(style_latent_space_mu.size(0), -1)
#             style_latent_space_logvar = style_latent_space_logvar.view(style_latent_space_logvar.size(0), -1)
#             class_latent_space_mu = class_latent_space_mu.view(class_latent_space_mu.size(0), -1)
#             class_latent_space_logvar = class_latent_space_logvar.view(class_latent_space_logvar.size(0), -1)
#             return style_latent_space_mu, style_latent_space_logvar, class_latent_space_mu, class_latent_space_logvar
#         else:
#             latent_space_mu = self.hidden_mu(h)
#             latent_space_logvar = self.hidden_logvar(h)
#             latent_space_mu = latent_space_mu.view(latent_space_mu.size(0), -1)
#             latent_space_logvar = latent_space_logvar.view(latent_space_logvar.size(0), -1)
#             return None, None, latent_space_mu, latent_space_logvar


# class DecoderImg(nn.Module):
#     def __init__(self, flags):
#         super(DecoderImg, self).__init__()
#         self.flags = flags
#         self.hidden_dim = 400
#         modules = []
#         if flags.factorized_representation:
#             modules.append(nn.Sequential(nn.Linear(flags.style_mnist_dim+flags.class_dim, self.hidden_dim), nn.ReLU(True)))
#         else:
#             modules.append(nn.Sequential(nn.Linear(flags.class_dim, self.hidden_dim), nn.ReLU(True)))

#         modules.extend([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(True))
#                         for _ in range(flags.num_hidden_layers - 1)])
#         self.dec = nn.Sequential(*modules)
#         self.fc3 = nn.Linear(self.hidden_dim, 784)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, style_latent_space, class_latent_space):

#         if self.flags.factorized_representation:
#             z = torch.cat((style_latent_space, class_latent_space), dim=1)
#         else:
#             z = class_latent_space
#         x_hat = self.dec(z)
#         x_hat = self.fc3(x_hat)
#         x_hat = self.sigmoid(x_hat)
#         x_hat = x_hat.view(*z.size()[:-1], *dataSize)
#         return x_hat, torch.tensor(0.75).to(z.device)

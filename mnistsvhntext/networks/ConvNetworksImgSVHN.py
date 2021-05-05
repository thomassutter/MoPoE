
import torch
import torch.nn as nn

class EncoderSVHN(nn.Module):
    def __init__(self, flags):
        super(EncoderSVHN, self).__init__()
        self.flags = flags;
        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1, dilation=1);
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, dilation=1);
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, dilation=1);
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0, dilation=1);
        self.relu = nn.ReLU();

        if flags.factorized_representation:
            # style
            self.style_mu = nn.Linear(in_features=128, out_features=flags.style_svhn_dim, bias=True)
            self.style_logvar = nn.Linear(in_features=128, out_features=flags.style_svhn_dim, bias=True)
            # class
            self.class_mu = nn.Linear(in_features=128, out_features=flags.class_dim, bias=True)
            self.class_logvar = nn.Linear(in_features=128, out_features=flags.class_dim, bias=True)
        else:
            #non-factorized
            self.hidden_mu = nn.Linear(in_features=128, out_features=flags.class_dim, bias=True)
            self.hidden_logvar = nn.Linear(in_features=128, out_features=flags.class_dim, bias=True)


    def forward(self, x):
        h = self.conv1(x);
        h = self.relu(h);
        h = self.conv2(h);
        h = self.relu(h);
        h = self.conv3(h);
        h = self.relu(h);
        h = self.conv4(h);
        h = self.relu(h);
        h = h.view(h.size(0), -1);
        if self.flags.factorized_representation:
            style_latent_space_mu = self.style_mu(h)
            style_latent_space_logvar = self.style_logvar(h)
            class_latent_space_mu = self.class_mu(h)
            class_latent_space_logvar = self.class_logvar(h)
            style_latent_space_mu = style_latent_space_mu.view(style_latent_space_mu.size(0), -1);
            style_latent_space_logvar = style_latent_space_logvar.view(style_latent_space_logvar.size(0), -1);
            class_latent_space_mu = class_latent_space_mu.view(class_latent_space_mu.size(0), -1);
            class_latent_space_logvar = class_latent_space_logvar.view(class_latent_space_logvar.size(0), -1);
            return style_latent_space_mu, style_latent_space_logvar, class_latent_space_mu, class_latent_space_logvar;
        else:
            latent_space_mu = self.hidden_mu(h);
            latent_space_logvar = self.hidden_logvar(h);
            latent_space_mu = latent_space_mu.view(latent_space_mu.size(0), -1);
            latent_space_logvar = latent_space_logvar.view(latent_space_logvar.size(0), -1);
            return None, None, latent_space_mu, latent_space_logvar;


class DecoderSVHN(nn.Module):
    def __init__(self, flags):
        super(DecoderSVHN, self).__init__();
        self.flags = flags;
        if flags.factorized_representation:
            self.linear_factorized = nn.Linear(flags.style_svhn_dim+flags.class_dim, 128);
        else:
            self.linear = nn.Linear(flags.class_dim, 128);
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=1, padding=0, dilation=1);
        self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, dilation=1);
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, dilation=1);
        self.conv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1, dilation=1);
        self.relu = nn.ReLU();

    def forward(self, style_latent_space, class_latent_space):
        if self.flags.factorized_representation:
            z = torch.cat((style_latent_space, class_latent_space), dim=1)
            z = self.linear_factorized(z)
        else:
            z = self.linear(class_latent_space)
        z = z.view(z.size(0), z.size(1), 1, 1);
        x_hat = self.relu(z);
        x_hat = self.conv1(x_hat);
        x_hat = self.relu(x_hat);
        x_hat = self.conv2(x_hat);
        x_hat = self.relu(x_hat);
        x_hat = self.conv3(x_hat);
        x_hat = self.relu(x_hat);
        x_hat = self.conv4(x_hat);
        return x_hat, torch.tensor(0.75).to(z.device);


import torch
import torch.nn as nn

from mnistsvhntext.networks.ConvNetworksTextMNIST import FeatureEncText

# Residual block
class ResidualBlockEncoder(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, downsample):
        super(ResidualBlockEncoder, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels_in);
        self.conv1 = nn.Conv1d(channels_in, channels_in, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(channels_in);
        self.conv2 = nn.Conv1d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, dilation=dilation)
        self.downsample = downsample;

    def forward(self, x):
        residual = x;
        out = self.bn1(x)
        out = self.relu(out);
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x);
        out = residual + 0.3*out
        return out



class ResidualBlockDecoder(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, upsample):
        super(ResidualBlockDecoder, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels_in);
        self.conv1 = nn.ConvTranspose1d(channels_in, channels_in, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(channels_in);
        self.conv2 = nn.ConvTranspose1d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, dilation=dilation, output_padding=1)
        self.upsample = upsample;


    def forward(self, x):
        residual = x;
        out = self.bn1(x)
        out = self.relu(out);
        out = self.conv1(out);
        out = self.bn2(out);
        out = self.relu(out);
        out = self.conv2(out);
        if self.upsample:
            residual = self.upsample(x);
        out = 2.0*residual + 0.3*out
        return out


def make_res_block_encoder(channels_in, channels_out, kernelsize, stride, padding, dilation):
    downsample = None;
    if (stride != 1) or (channels_in != channels_out) or dilation != 1:
        downsample = nn.Sequential(nn.Conv1d(channels_in, channels_out,
                                             kernel_size=kernelsize,
                                             stride=stride,
                                             padding=padding,
                                             dilation=dilation),
                                   nn.BatchNorm1d(channels_out))
    layers = []
    layers.append(ResidualBlockEncoder(channels_in, channels_out, kernelsize, stride, padding, dilation, downsample))
    return nn.Sequential(*layers)


def make_res_block_decoder(channels_in, channels_out, kernelsize, stride, padding, dilation):
    upsample = None;
    if (kernelsize != 1 or stride != 1) or (channels_in != channels_out) or dilation != 1:
        upsample = nn.Sequential(nn.ConvTranspose1d(channels_in, channels_out,
                                                    kernel_size=kernelsize,
                                                    stride=stride,
                                                    padding=padding,
                                                    dilation=dilation,
                                                    output_padding=1),
                                   nn.BatchNorm1d(channels_out))
    layers = []
    layers.append(ResidualBlockDecoder(channels_in, channels_out, kernelsize, stride, padding, dilation, upsample))
    return nn.Sequential(*layers)


class ClfText(nn.Module):
    def __init__(self, flags):
        super(ClfText, self).__init__()
        self.flags = flags
        self.conv1 = nn.Conv1d(flags.num_features, 2 * flags.dim, kernel_size=1);
        self.resblock_1 = make_res_block_encoder(2 * flags.dim, 3 * flags.dim, kernelsize=4, stride=2, padding=1,
                                                 dilation=1);
        self.resblock_4 = make_res_block_encoder(3 * flags.dim, 2 * flags.dim, kernelsize=4, stride=2, padding=0,
                                                 dilation=1);
        self.dropout = nn.Dropout(p=0.5, inplace=False);
        self.linear = nn.Linear(in_features=2*flags.dim, out_features=10, bias=True) # 10 is the number of classes
        self.sigmoid = nn.Sigmoid();


    def forward(self, x):
        x = x.transpose(-2,-1)
        h = self.conv1(x);
        h = self.resblock_1(h);
        h = self.resblock_4(h);
        h = self.dropout(h);
        h = h.view(h.size(0), -1);
        h = self.linear(h);
        out = self.sigmoid(h);
        return out;


    def get_activations(self, x):
        h = self.conv1(x);
        h = self.resblock_1(h);
        h = self.resblock_2(h);
        h = self.resblock_3(h);
        h = self.resblock_4(h);
        h = h.view(h.size(0), -1);
        return h;

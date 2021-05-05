
import numpy as np

import torch.nn as nn


from celeba.networks.ResidualBlocks import ResidualBlock1dConv


def make_res_block_encoder_feature_compressor(channels_in, channels_out, a_val=2, b_val=0.3):
    downsample = None;
    if channels_in != channels_out:
        downsample = nn.Sequential(nn.Conv1d(channels_in,
                                             channels_out,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             dilation=1),
                                   nn.BatchNorm1d(channels_out))
    layers = []
    layers.append(ResidualBlock1dConv(channels_in, channels_out, kernelsize=1, stride=1, padding=0, dilation=1, downsample=downsample, a=a_val, b=b_val))
    return nn.Sequential(*layers)


def make_layers_resnet_encoder_feature_compressor(start_channels, end_channels, a=2, b=0.3, l=1):
    layers = [];
    num_compr_layers = int((1/float(l))*np.floor(np.log(start_channels / float(end_channels))))
    for k in range(0, num_compr_layers):
        in_channels = np.round(start_channels/float(2 ** (l*k))).astype(int)
        out_channels = np.round(start_channels/float(2 ** (l*(k+1)))).astype(int)
        resblock = make_res_block_encoder_feature_compressor(in_channels, out_channels, a_val=a, b_val=b);
        layers.append(resblock);

    out_channels = np.round(start_channels/float(2 ** (l*num_compr_layers))).astype(int)
    if out_channels > end_channels:
        resblock = make_res_block_encoder_feature_compressor(out_channels, end_channels, a_val=a, b_val=b);
        layers.append(resblock)
    return nn.Sequential(*layers)


class ResidualFeatureCompressor(nn.Module):
    def __init__(self, in_channels, out_channels_style, out_channels_content, a, b, compression_power):
        super(ResidualFeatureCompressor, self).__init__()
        self.a = a;
        self.b = b;
        self.compression_power = compression_power
        self.style_mu = make_res_block_encoder_feature_compressor(in_channels, out_channels_style, a_val=self.a, b_val=self.b)
        self.style_logvar = make_res_block_encoder_feature_compressor(in_channels, out_channels_style, a_val=self.a, b_val=self.b)
        self.content_mu = make_res_block_encoder_feature_compressor(in_channels, out_channels_content, a_val=self.a, b_val=self.b)
        self.content_logvar = make_res_block_encoder_feature_compressor(in_channels, out_channels_content, a_val=self.a, b_val=self.b)

    def forward(self, feats):
        mu_style, logvar_style = self.style_mu(feats), self.style_logvar(feats);
        mu_content, logvar_content = self.content_mu(feats), self.content_logvar(feats);
        return mu_style, logvar_style, mu_content, logvar_content;

class LinearFeatureCompressor(nn.Module):
    def __init__(self, in_channels, out_channels_style, out_channels_content):
        super(LinearFeatureCompressor, self).__init__();
        self.style_mu = nn.Linear(in_channels, out_channels_style, bias=True);
        self.style_logvar = nn.Linear(in_channels, out_channels_style, bias=True);
        self.content_mu = nn.Linear(in_channels, out_channels_content, bias=True);
        self.content_logvar = nn.Linear(in_channels, out_channels_content, bias=True);

    def forward(self, feats):
        feats = feats.view(feats.size(0), -1);
        mu_style, logvar_style = self.style_mu(feats), self.style_logvar(feats);
        mu_content, logvar_content = self.content_mu(feats), self.content_logvar(feats);
        return mu_style, logvar_style, mu_content, logvar_content;

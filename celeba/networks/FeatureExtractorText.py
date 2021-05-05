import torch.nn as nn

from celeba.networks.ResidualBlocks import ResidualBlock1dConv


def make_res_block_encoder_feature_extractor(in_channels, out_channels, kernelsize, stride, padding, dilation, a_val=2.0, b_val=0.3):
    downsample = None;
    if (stride != 1) or (in_channels != out_channels) or dilation != 1:
        downsample = nn.Sequential(nn.Conv1d(in_channels, out_channels,
                                             kernel_size=kernelsize,
                                             stride=stride,
                                             padding=padding,
                                             dilation=dilation),
                                   nn.BatchNorm1d(out_channels))
    layers = []
    layers.append(ResidualBlock1dConv(in_channels, out_channels, kernelsize, stride, padding, dilation, downsample, a=a_val, b=b_val))
    return nn.Sequential(*layers)


class FeatureExtractorText(nn.Module):
    def __init__(self, args, a, b):
        super(FeatureExtractorText, self).__init__()
        self.args = args
        self.a = a
        self.b = b
        self.conv1 = nn.Conv1d(self.args.num_features, self.args.DIM_text,
                               kernel_size=4, stride=2, padding=1, dilation=1);
        self.resblock_1 = make_res_block_encoder_feature_extractor(self.args.DIM_text, 2*self.args.DIM_text,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_2 = make_res_block_encoder_feature_extractor(2*self.args.DIM_text, 3*self.args.DIM_text,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_3 = make_res_block_encoder_feature_extractor(3*self.args.DIM_text, 4*self.args.DIM_text,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_4 = make_res_block_encoder_feature_extractor(4*self.args.DIM_text, 5*self.args.DIM_text,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_5 = make_res_block_encoder_feature_extractor(5*self.args.DIM_text, 5*self.args.DIM_text,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_6 = make_res_block_encoder_feature_extractor(5*self.args.DIM_text, 5*self.args.DIM_text,
                                                                   kernelsize=4, stride=2, padding=0, dilation=1);

    def forward(self, x):
        x = x.transpose(-2,-1);
        out = self.conv1(x)
        out = self.resblock_1(out);
        out = self.resblock_2(out);
        out = self.resblock_3(out);
        out = self.resblock_4(out);
        out = self.resblock_5(out);
        out = self.resblock_6(out);
        return out

import torch.nn as nn

from celeba.networks.ResidualBlocks import ResidualBlock1dTransposeConv


def res_block_decoder(in_channels, out_channels, kernelsize, stride, padding, o_padding, dilation, a_val=2.0, b_val=0.3):
    upsample = None;

    if (kernelsize != 1 or stride != 1) or (in_channels != out_channels) or dilation != 1:
        upsample = nn.Sequential(nn.ConvTranspose1d(in_channels, out_channels,
                                                    kernel_size=kernelsize,
                                                    stride=stride,
                                                    padding=padding,
                                                    dilation=dilation,
                                                    output_padding=o_padding),
                                   nn.BatchNorm1d(out_channels))
    layers = []
    layers.append(ResidualBlock1dTransposeConv(in_channels, out_channels, kernelsize, stride, padding, dilation, o_padding, upsample=upsample, a=a_val, b=b_val))
    return nn.Sequential(*layers)


class DataGeneratorText(nn.Module):
    def __init__(self, args, a, b):
        super(DataGeneratorText, self).__init__()
        self.args = args
        self.a = a;
        self.b = b;
        self.resblock_1 = res_block_decoder(5*args.DIM_text, 5*args.DIM_text,
                                            kernelsize=4, stride=1, padding=0, dilation=1, o_padding=0);
        self.resblock_2 = res_block_decoder(5*args.DIM_text, 5*args.DIM_text,
                                            kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0);
        self.resblock_3 = res_block_decoder(5*args.DIM_text, 4*args.DIM_text,
                                            kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0);
        self.resblock_4 = res_block_decoder(4*args.DIM_text, 3*args.DIM_text,
                                            kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0);
        self.resblock_5 = res_block_decoder(3*args.DIM_text, 2*args.DIM_text,
                                            kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0);
        self.resblock_6 = res_block_decoder(2*args.DIM_text, args.DIM_text,
                                                 kernelsize=4, stride=2, padding=1, dilation=1, o_padding=0);
        self.conv2 = nn.ConvTranspose1d(self.args.DIM_text, args.num_features,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        dilation=1,
                                        output_padding=0);
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, feats):
        d = self.resblock_1(feats);
        d = self.resblock_2(d);
        d = self.resblock_3(d);
        d = self.resblock_4(d);
        d = self.resblock_5(d);
        d = self.resblock_6(d);
        d = self.conv2(d)
        d = self.softmax(d);
        return d

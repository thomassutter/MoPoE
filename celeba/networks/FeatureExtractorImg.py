import torch.nn as nn

from celeba.networks.ResidualBlocks import ResidualBlock2dConv


def make_res_block_feature_extractor(in_channels, out_channels, kernelsize, stride, padding, dilation, a_val=2.0, b_val=0.3):
    downsample = None;
    if (stride != 2) or (in_channels != out_channels):
        downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                             kernel_size=kernelsize,
                                             padding=padding,
                                             stride=stride,
                                             dilation=dilation),
                                   nn.BatchNorm2d(out_channels))
    layers = [];
    layers.append(ResidualBlock2dConv(in_channels, out_channels, kernelsize, stride, padding, dilation, downsample,a=a_val, b=b_val))
    return nn.Sequential(*layers)


class FeatureExtractorImg(nn.Module):
    def __init__(self, args, a, b):
        super(FeatureExtractorImg, self).__init__();
        self.args = args;
        self.a = a;
        self.b = b;
        self.conv1 = nn.Conv2d(self.args.image_channels, self.args.DIM_img,
                              kernel_size=3,
                              stride=2,
                              padding=2,
                              dilation=1,
                              bias=False)
        self.resblock1 = make_res_block_feature_extractor(args.DIM_img, 2 * args.DIM_img, kernelsize=4, stride=2,
                                                          padding=1, dilation=1, a_val=a, b_val=b)
        self.resblock2 = make_res_block_feature_extractor(2 * args.DIM_img, 3 * args.DIM_img, kernelsize=4, stride=2,
                                                          padding=1, dilation=1, a_val=self.a, b_val=self.b)
        self.resblock3 = make_res_block_feature_extractor(3 * args.DIM_img, 4 * args.DIM_img, kernelsize=4, stride=2,
                                                          padding=1, dilation=1, a_val=self.a, b_val=self.b)
        self.resblock4 = make_res_block_feature_extractor(4 * args.DIM_img, 5 * args.DIM_img, kernelsize=4, stride=2,
                                                          padding=0, dilation=1, a_val=self.a, b_val=self.b)

    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock1(out);
        out = self.resblock2(out);
        out = self.resblock3(out);
        out = self.resblock4(out);
        return out

import torch
import torch.nn as nn



# Residual block
class ResidualBlock1dConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, downsample, a=2, b=0.3):
        super(ResidualBlock1dConv, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels_in);
        self.conv1 = nn.Conv1d(channels_in, channels_in, kernel_size=1, stride=1, padding=0)
        self.dropout1 = nn.Dropout(p=0.5, inplace=False);
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(channels_in);
        self.conv2 = nn.Conv1d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, dilation=dilation)
        self.dropout2 = nn.Dropout(p=0.5, inplace=False);
        self.downsample = downsample;
        self.a = a;
        self.b = b;

    def forward(self, x):
        residual = x;
        out = self.bn1(x)
        out = self.relu(out);
        out = self.conv1(out)
        out = self.dropout1(out);
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out);
        if self.downsample:
            residual = self.downsample(x);
        out = self.a*residual + self.b*out
        return out


class ResidualBlock1dTransposeConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, o_padding, upsample, a=2, b=0.3):
        super(ResidualBlock1dTransposeConv, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels_in);
        self.conv1 = nn.ConvTranspose1d(channels_in, channels_in, kernel_size=1, stride=1, padding=0)
        self.dropout1 = nn.Dropout(p=0.5, inplace=False);
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(channels_in);
        self.conv2 = nn.ConvTranspose1d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, dilation=dilation, output_padding=o_padding)
        self.dropout2 = nn.Dropout(p=0.5, inplace=False);
        self.upsample = upsample;
        self.a = a;
        self.b = b;


    def forward(self, x):
        residual = x;
        out = self.bn1(x)
        out = self.relu(out);
        out = self.conv1(out);
        out = self.dropout1(out);
        out = self.bn2(out);
        out = self.relu(out);
        out = self.conv2(out);
        out = self.dropout2(out);
        if self.upsample:
            residual = self.upsample(x);
        out = self.a*residual + self.b*out
        return out


class ResidualBlock2dConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, downsample, a=1, b=1):
        super(ResidualBlock2dConv, self).__init__();
        self.conv1 = nn.Conv2d(channels_in, channels_in, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False)
        self.dropout1 = nn.Dropout2d(p=0.5, inplace=False)
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(channels_in)
        self.conv2 = nn.Conv2d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.dropout2 = nn.Dropout2d(p=0.5, inplace=False)
        self.downsample = downsample
        self.a = a;
        self.b = b;

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.a*residual + self.b*out;
        return out


class ResidualBlock2dTransposeConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, o_padding, upsample, a=1, b=1):
        super(ResidualBlock2dTransposeConv, self).__init__();
        self.conv1 = nn.ConvTranspose2d(channels_in, channels_in, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False)
        self.dropout1 = nn.Dropout2d(p=0.5, inplace=False);
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(channels_in)
        self.conv2 = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, dilation=dilation, bias=False, output_padding=o_padding)
        self.dropout2 = nn.Dropout2d(p=0.5, inplace=False);
        # self.conv3 = nn.ConvTranspose2d(channels_out, channels_out, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False)
        # self.bn3 = nn.BatchNorm2d(channels_out)
        self.upsample = upsample
        self.a = a;
        self.b = b;

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        if self.upsample is not None:
            residual = self.upsample(x)
        out = self.a * residual + self.b * out;
        return out
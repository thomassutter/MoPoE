import torch
import torch.nn as nn

from celeba.networks.FeatureExtractorImg import FeatureExtractorImg


class ClfImg(nn.Module):
    def __init__(self, flags):
        super(ClfImg, self).__init__();
        self.feature_extractor = FeatureExtractorImg(flags, a=2.0, b=0.3);
        self.dropout = nn.Dropout(p=0.5, inplace=False);
        self.linear = nn.Linear(in_features=flags.num_layers_img*flags.DIM_img, out_features=40, bias=True);
        self.sigmoid = nn.Sigmoid();

    def forward(self, x_img):
        h = self.feature_extractor(x_img);
        h = self.dropout(h);
        h = h.view(h.size(0), -1);
        h = self.linear(h);
        out = self.sigmoid(h)
        return out;

    def get_activations(self, x_img):
        h = self.feature_extractor(x_img);
        return h;

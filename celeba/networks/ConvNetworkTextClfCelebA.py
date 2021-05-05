
import torch.nn as nn

from celeba.networks.FeatureExtractorText import FeatureExtractorText
from celeba.networks.FeatureExtractorText import make_res_block_encoder_feature_extractor


class ClfText(nn.Module):
    def __init__(self, flags):
        super(ClfText, self).__init__();
        self.args = flags; 
        self.conv1 = nn.Conv1d(self.args.num_features, self.args.DIM_text,
                               kernel_size=3, stride=2, padding=1, dilation=1);
        self.resblock_1 = make_res_block_encoder_feature_extractor(self.args.DIM_text, 2*self.args.DIM_text,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_2 = make_res_block_encoder_feature_extractor(2*self.args.DIM_text, 3*self.args.DIM_text,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_3 = make_res_block_encoder_feature_extractor(3*self.args.DIM_text, 4*self.args.DIM_text,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_4 = make_res_block_encoder_feature_extractor(4*self.args.DIM_text, 5*self.args.DIM_text,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_5 = make_res_block_encoder_feature_extractor(5*self.args.DIM_text, 6*self.args.DIM_text,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_6 = make_res_block_encoder_feature_extractor(6*self.args.DIM_text, 7*self.args.DIM_text,
                                                                   kernelsize=4, stride=2, padding=0, dilation=1);
        self.dropout = nn.Dropout(p=0.5, inplace=False);
        self.linear = nn.Linear(in_features=flags.num_layers_text*flags.DIM_text, out_features=40, bias=True)
        self.sigmoid = nn.Sigmoid();


    def forward(self, x_text):
        x_text = x_text.transpose(-2,-1);
        out = self.conv1(x_text)
        out = self.resblock_1(out);
        out = self.resblock_2(out);
        out = self.resblock_3(out);
        out = self.resblock_4(out);
        out = self.resblock_5(out);
        out = self.resblock_6(out);
        h = self.dropout(out);
        h = h.view(h.size(0), -1);
        h = self.linear(h);
        out = self.sigmoid(h)
        return out;

import torch
import torch.nn as nn

from celeba.networks.FeatureExtractorImg import FeatureExtractorImg
from celeba.networks.FeatureCompressor import LinearFeatureCompressor
from celeba.networks.DataGeneratorImg import DataGeneratorImg

class EncoderImg(nn.Module):
    def __init__(self, flags):
        super(EncoderImg, self).__init__();
        self.feature_extractor = FeatureExtractorImg(flags, a=2.0, b=0.3)
        self.feature_compressor = LinearFeatureCompressor(flags.num_layers_img*flags.DIM_img,
                                                          flags.style_img_dim,
                                                          flags.class_dim)

    def forward(self, x_img):
        h_img = self.feature_extractor(x_img);
        h_img = h_img.view(h_img.shape[0], h_img.shape[1], h_img.shape[2])
        mu_style, logvar_style, mu_content, logvar_content = self.feature_compressor(h_img);
        return mu_style, logvar_style, mu_content, logvar_content, h_img;


class DecoderImg(nn.Module):
    def __init__(self, flags):
        super(DecoderImg, self).__init__();
        self.feature_generator = nn.Linear(flags.style_img_dim + flags.class_dim, flags.num_layers_img * flags.DIM_img, bias=True);
        self.img_generator = DataGeneratorImg(flags, a=2.0, b=0.3)

    def forward(self, z_style, z_content):
        z = torch.cat((z_style, z_content), dim=1).squeeze(-1)
        img_feat_hat = self.feature_generator(z);
        img_feat_hat = img_feat_hat.view(img_feat_hat.size(0), img_feat_hat.size(1), 1, 1);
        img_hat = self.img_generator(img_feat_hat)
        return img_hat, torch.tensor(0.75).to(z.device);

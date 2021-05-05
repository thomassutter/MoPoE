

import torch

from modalities.Modality import Modality

from utils import utils
from utils.save_samples import write_samples_img_to_file


class MimicPA(Modality):
    def __init__(self, name, enc, dec, class_dim, style_dim, lhood_name):
        super().__init__(name, enc, dec, class_dim, style_dim, lhood_name);
        self.data_size = torch.Size((1, 128, 128));
        self.gen_quality_eval = True;
        self.file_suffix = '.png';


    def save_data(self, d, fn, args):
        img_per_row = args['img_per_row'];
        write_samples_img_to_file(d, fn, img_per_row);


    def plot_data(self, d):
        p = d.repeat(1, 1, 1, 1);
        return p;

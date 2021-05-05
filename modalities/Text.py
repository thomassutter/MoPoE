
import torch

from modalities.Modality import Modality

from utils import utils
from utils import plot
from utils.save_samples import write_samples_text_to_file
from utils.text import tensor_to_text


class Text(Modality):
    def __init__(self, name, enc, dec, class_dim, style_dim, lhood_name,
                 len_sequence, alphabet, plotImgSize, font):
        super().__init__(name, enc, dec, class_dim, style_dim, lhood_name)
        self.alphabet = alphabet;
        self.len_sequence = len_sequence;
        self.data_size = torch.Size([len_sequence]);
        self.plot_img_size = plotImgSize;
        self.font = font;
        self.gen_quality_eval = False;
        self.file_suffix = '.txt';


    def save_data(self, d, fn, args):
        write_samples_text_to_file(tensor_to_text(self.alphabet,
                                                  d.unsqueeze(0)),
                                   fn);

 
    def plot_data(self, d):
        out = plot.text_to_pil(d.unsqueeze(0), self.plot_img_size,
                               self.alphabet, self.font)
        return out;

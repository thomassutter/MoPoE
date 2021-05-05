

import torch

from modalities.Modality import Modality

from utils import utils
from utils import plot
from utils.save_samples import write_samples_text_to_file
from utils.text import tensor_to_text


class Text(Modality):
    def __init__(self, enc, dec, len_sequence, alphabet, plotImgSize, font):
        self.name = 'text';
        self.likelihood_name = 'categorical';
        self.alphabet = alphabet;
        self.len_sequence = len_sequence;
        #self.data_size = torch.Size((len(alphabet), len_sequence));
        self.data_size = torch.Size([len_sequence]);
        self.plot_img_size = plotImgSize;
        self.font = font;
        self.gen_quality_eval = False;
        self.file_suffix = '.txt';
        self.encoder = enc;
        self.decoder = dec;
        self.likelihood = utils.get_likelihood(self.likelihood_name);


    def save_data(self, d, fn, args):
        write_samples_text_to_file(tensor_to_text(self.alphabet,
                                                  d.unsqueeze(0)),
                                   fn);

 
    def plot_data(self, d):
        out = plot.text_to_pil(d.unsqueeze(0), self.plot_img_size,
                               self.alphabet, self.font,
                              w=256, h=256, linewidth=16)
        return out;

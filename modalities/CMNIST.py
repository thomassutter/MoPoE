
import torch
from torchvision import transforms

from PIL import Image

from modalities.Modality import Modality

from utils import utils
from utils.save_samples import write_samples_img_to_file


class CMNIST(Modality):
    def __init__(self, name, enc, dec, class_dim, style_dim, lhood_name):
        super().__init__(name, enc, dec, class_dim, style_dim, lhood_name);
        self.data_size = torch.Size((3, 28, 28))
        self.gen_quality_eval = True
        self.file_suffix = '.png'
        # self.transform = transforms.Compose([transforms.ToTensor()])

    def save_data(self, d, fn, args):
        img_per_row = args['img_per_row']
        write_samples_img_to_file(d, fn, img_per_row)

    def plot_data(self, d):
        # out = self.transform(d.squeeze(0).cpu()).cuda().unsqueeze(0)
        # return out
        return d


import torch
from torchvision import transforms

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from modalities.Modality import Modality

from utils import utils
from utils.save_samples import write_samples_img_to_file


class SVHN(Modality):
    def __init__(self, name, enc, dec, class_dim, style_dim, lhood_name,
                 plotImgSize):
        super().__init__(name, enc, dec, class_dim, style_dim, lhood_name);
        self.data_size = torch.Size((3, 32, 32));
        self.plot_img_size = plotImgSize;
        self.transform_plot = self.get_plot_transform();
        self.gen_quality_eval = True;
        self.file_suffix = '.png';


    def save_data(self, d, fn, args):
        img_per_row = args['img_per_row'];
        write_samples_img_to_file(d, fn, img_per_row);

 
    def plot_data(self, d):
        out = self.transform_plot(d.squeeze(0).cpu()).cuda().unsqueeze(0);
        return out;


    def get_plot_transform(self):
        transf = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize(size=list(self.plot_img_size)[1:],
                                                       interpolation=Image.BICUBIC),
                                     transforms.ToTensor()])
        return transf;

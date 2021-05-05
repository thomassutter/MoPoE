

import os 
import random
import numpy as np 
from itertools import chain, combinations

import PIL.Image as Image
from PIL import ImageFont 
import torch
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import average_precision_score

from modalities.CelebaImg import Img
from modalities.CelebaText import Text
from celeba.CelebADataset import CelebaDataset
from celeba.networks.VAEbimodalCelebA import VAEbimodalCelebA
from celeba.networks.ConvNetworkImgClfCelebA import ClfImg as ClfImg
from celeba.networks.ConvNetworkTextClfCelebA import ClfText as ClfText

from celeba.networks.ConvNetworksImgCelebA import EncoderImg, DecoderImg
from celeba.networks.ConvNetworksTextCelebA import EncoderText, DecoderText

from utils.BaseExperiment import BaseExperiment


LABELS = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
          'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
          'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
          'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
          'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
          'Mouth_Slightly_Open', 'Mustache',
          'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
          'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
          'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
          'Wearing_Earrings', 'Wearing_Hat',
          'Wearing_Lipstick', 'Wearing_Necklace',
          'Wearing_Necktie', 'Young'];

class CelebaExperiment(BaseExperiment):
    def __init__(self, flags, alphabet):
        self.labels = LABELS;
        self.flags = flags;
        self.dataset = flags.dataset;
        self.plot_img_size = torch.Size((3, 64, 64))
        self.font = ImageFont.truetype('FreeSerif.ttf', 38)

        self.alphabet = alphabet;
        self.flags.num_features = len(alphabet);

        self.modalities = self.set_modalities();
        self.num_modalities = len(self.modalities.keys());
        self.subsets = self.set_subsets();
        self.dataset_train = None;
        self.dataset_test = None;
        self.set_dataset();

        self.mm_vae = self.set_model();
        self.clfs = self.set_clfs();
        self.optimizer = None;
        self.rec_weights = self.set_rec_weights();
        self.style_weights = self.set_style_weights();

        self.test_samples = self.get_test_samples();
        self.eval_metric = average_precision_score; 
        self.paths_fid = self.set_paths_fid();


    def set_model(self):
        model = VAEbimodalCelebA(self.flags, self.modalities, self.subsets)
        model = model.to(self.flags.device);
        return model;


    def set_modalities(self):
        mod1 = Img(EncoderImg(self.flags),
                   DecoderImg(self.flags),
                   self.plot_img_size);
        mod2 = Text(EncoderText(self.flags),
                    DecoderText(self.flags),
                    self.flags.len_sequence,
                    self.alphabet,
                    self.plot_img_size,
                    self.font);
        mods = {mod1.name: mod1, mod2.name: mod2};
        return mods;


    def get_transform_celeba(self):
        offset_height = (218 - self.flags.crop_size_img) // 2
        offset_width = (178 - self.flags.crop_size_img) // 2
        crop = lambda x: x[:, offset_height:offset_height + self.flags.crop_size_img,
                         offset_width:offset_width + self.flags.crop_size_img]
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(crop),
                                        transforms.ToPILImage(),
                                        transforms.Resize(size=(self.flags.img_size,
                                                                self.flags.img_size),
                                                          interpolation=Image.BICUBIC),
                                        transforms.ToTensor()])

        return transform;

    def set_dataset(self):
        transform = self.get_transform_celeba();
        d_train = CelebaDataset(self.flags, self.alphabet, partition=0, transform=transform)
        d_eval = CelebaDataset(self.flags, self.alphabet, partition=1, transform=transform)
        self.dataset_train = d_train;
        self.dataset_test = d_eval;


    def set_clfs(self):
        model_clf_m1 = None;
        model_clf_m2 = None;
        if self.flags.use_clf:
            model_clf_m1 = ClfImg(self.flags);
            model_clf_m1.load_state_dict(torch.load(os.path.join(self.flags.dir_clf,
                                                                 self.flags.clf_save_m1)))
            model_clf_m1 = model_clf_m1.to(self.flags.device);

            model_clf_m2 = ClfText(self.flags);
            model_clf_m2.load_state_dict(torch.load(os.path.join(self.flags.dir_clf,
                                                                 self.flags.clf_save_m2)))
            model_clf_m2 = model_clf_m2.to(self.flags.device);

        clfs = {'img': model_clf_m1,
                'text': model_clf_m2}
        return clfs;


    def set_optimizer(self):
        # optimizer definition
        optimizer = optim.Adam(
            list(self.mm_vae.parameters()),
            lr=self.flags.initial_learning_rate,
            betas=(self.flags.beta_1, self.flags.beta_2))
        self.optimizer = optimizer;


    def set_rec_weights(self):
        rec_weights = dict();
        ref_mod_d_size = self.modalities['img'].data_size.numel()/3;
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key];
            numel_mod = mod.data_size.numel()
            rec_weights[mod.name] = float(ref_mod_d_size/numel_mod)
        return rec_weights;


    def set_style_weights(self):
        weights = dict();
        weights['img'] = self.flags.beta_m1_style;
        weights['text'] = self.flags.beta_m2_style;
        return weights;


    def get_prediction_from_attr(self, values):
        return values.ravel();


    def get_prediction_from_attr_random(self, values, index=None):
        return values[:,index] > 0.5;


    def eval_label(self, values, labels, index=None):
        pred = values[:,index];
        gt = labels[:,index];
        try:
            ap = self.eval_metric(gt, pred) 
        except ValueError:
            ap = 0.0;
        return ap;


    def get_test_samples(self, num_images=10):
        n_test = self.dataset_test.__len__();
        samples = []
        for i in range(10):
            ix = np.random.randint(0, len(self.dataset_test.img_names))
            sample, target = self.dataset_test.__getitem__(random.randint(0, n_test))
            for k, key in enumerate(sample):
                sample[key] = sample[key].to(self.flags.device);
            samples.append(sample)
        return samples


    def mean_eval_metric(self, values):
        return np.mean(np.array(values));




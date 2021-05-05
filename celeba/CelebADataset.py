import os

import pandas as pd
from torch.utils.data import Dataset
import PIL.Image as Image

import torch

from utils import text as text


class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, args, alphabet, partition=0, transform=None):

        self.dir_dataset_base = os.path.join(args.dir_data, args.dataset)

        filename_text = os.path.join(args.dir_text, 'list_attr_text_' + str(args.len_sequence).zfill(3) + '_' + str(args.random_text_ordering) + '_' + str(args.random_text_startindex) + '_celeba.csv');
        filename_partition = os.path.join(self.dir_dataset_base, 'list_eval_partition.csv');
        filename_attributes = os.path.join(self.dir_dataset_base, 'list_attr_celeba.csv')

        df_text = pd.read_csv(filename_text)
        df_partition = pd.read_csv(filename_partition);
        df_attributes = pd.read_csv(filename_attributes);

        self.args = args;
        self.img_dir = os.path.join(self.dir_dataset_base, 'img_align_celeba');
        self.txt_path = filename_text
        self.attrributes_path = filename_attributes;
        self.partition_path = filename_partition;

        self.alphabet = alphabet;
        self.img_names = df_text.loc[df_partition['partition'] == partition]['image_id'].values
        self.attributes = df_attributes.loc[df_partition['partition'] == partition];
        self.labels = df_attributes.loc[df_partition['partition'] == partition].values; #atm, i am just using blond_hair as labels
        self.y = df_text.loc[df_partition['partition'] == partition]['text'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir, self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)
        text_str = text.one_hot_encode(self.args.len_sequence, self.alphabet, self.y[index])
        label = torch.from_numpy((self.labels[index,1:] > 0).astype(int)).float();
        sample = {'img': img, 'text': text_str}
        return sample, label

    def __len__(self):
        return self.y.shape[0]


    def get_text_str(self, index):
        return self.y[index];

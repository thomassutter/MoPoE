import sys,os
import numpy as np
import pandas as pd

import torch
from torchvision.utils import save_image


def append_list_to_list_linear(l1, l2):
    for k in range(0, len(l2)):
        if isinstance(l2[k], str):
            l1.append(l2[k]);
        else:
            l1.append(l2[k].item());
    return l1;

def write_samples_text_to_file(samples, filename):
    file_samples = open(filename, 'w');
    for k in range(0, len(samples)):
        file_samples.write(''.join(samples[k]) + '\n');
    file_samples.close();

def getText(samples):
    lines = []
    for k in range(0, len(samples)):
        lines.append(''.join(samples[k])[::-1])
    text = '\n\n'.join(lines)
    print(text)
    return text

def write_samples_img_to_file(samples, filename, img_per_row=1):
    save_image(samples.data.cpu(), filename, nrow=img_per_row);


def save_generated_samples_singlegroup(exp, batch_id, group_name, samples):
    dir_save = exp.paths_fid[group_name];
    for k, key in enumerate(samples.keys()):
        dir_f = os.path.join(dir_save, key);
        if not os.path.exists(dir_f):
            os.makedirs(dir_f);

    cnt_samples = batch_id * exp.flags.batch_size;
    for k in range(0, exp.flags.batch_size):
        for i, key in enumerate(samples.keys()):
            mod = exp.modalities[key];
            fn_out = os.path.join(dir_save, key, str(cnt_samples).zfill(6) +
                                  mod.file_suffix);
            mod.save_data(samples[key][k], fn_out, {'img_per_row': 1});
        cnt_samples += 1;

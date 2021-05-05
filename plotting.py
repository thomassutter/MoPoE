import sys
import os

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import utils
from utils import plot


def generate_plots(exp, epoch):
    plots = dict();
    if exp.flags.factorized_representation:
        # mnist to mnist: swapping content and style intra modal
        swapping_figs = generate_swapping_plot(exp, epoch)
        plots['swapping'] = swapping_figs;

    for k in range(len(exp.modalities.keys())):
        cond_k = generate_conditional_fig_M(exp, epoch, k+1)
        plots['cond_gen_' + str(k+1).zfill(2)] = cond_k;

    plots['random'] = generate_random_samples_plots(exp, epoch);
    return plots;


def generate_random_samples_plots(exp, epoch):
    model = exp.mm_vae;
    mods = exp.modalities;
    num_samples = 100;
    random_samples = model.generate(num_samples)
    random_plots = dict();
    for k, m_key_in in enumerate(mods.keys()):
        mod = mods[m_key_in];
        samples_mod = random_samples[m_key_in];
        rec = torch.zeros(exp.plot_img_size,
                          dtype=torch.float32).repeat(num_samples,1,1,1);
        for l in range(0, num_samples):
            rand_plot = mod.plot_data(samples_mod[l]);
            rec[l, :, :, :] = rand_plot;
        random_plots[m_key_in] = rec;

    for k, m_key in enumerate(mods.keys()):
        fn = os.path.join(exp.flags.dir_random_samples, 'random_epoch_' +
                             str(epoch).zfill(4) + '_' + m_key + '.png');
        mod_plot = random_plots[m_key];
        p = plot.create_fig(fn, mod_plot, 10, save_figure=exp.flags.save_figure);
        random_plots[m_key] = p;
    return random_plots;


def generate_swapping_plot(exp, epoch):
    model = exp.mm_vae;
    mods = exp.modalities;
    samples = exp.test_samples;
    swap_plots = dict();
    for k, m_key_in in enumerate(mods.keys()):
        mod_in = mods[m_key_in];
        for l, m_key_out in enumerate(mods.keys()):
            mod_out = mods[m_key_out];
            rec = torch.zeros(exp.plot_img_size,
                              dtype=torch.float32).repeat(121,1,1,1);
            rec = rec.to(exp.flags.device);
            for i in range(len(samples)):
                c_sample_in = mod_in.plot_data(samples[i][mod_in.name]);
                s_sample_out = mod_out.plot_data(samples[i][mod_out.name]);
                rec[i+1, :, :, :] = c_sample_in;
                rec[(i + 1) * 11, :, :, :] = s_sample_out;
            # style transfer
            for i in range(len(samples)):
                for j in range(len(samples)):
                    i_batch_s = {mod_out.name: samples[i][mod_out.name].unsqueeze(0)}
                    i_batch_c = {mod_in.name: samples[i][mod_in.name].unsqueeze(0)}
                    l_style = model.inference(i_batch_s,
                                              num_samples=1)
                    l_content = model.inference(i_batch_c,
                                                num_samples=1)
                    l_s_mod = l_style['modalities'][mod_out.name + '_style'];
                    l_c_mod = l_content['modalities'][mod_in.name];
                    s_emb = model.reparameterize(l_s_mod[0], l_s_mod[1]);
                    c_emb = model.reparameterize(l_c_mod[0], l_c_mod[1]);
                    style_emb = {mod_out.name: s_emb}
                    emb_swap = {'content': c_emb, 'style': style_emb};
                    swap_sample = model.generate_from_latents(emb_swap);
                    swap_out = mod_out.plot_data(swap_sample[mod_out.name].squeeze(0));
                    rec[(i+1) * 11 + (j+1), :, :, :] = swap_out;
                    fn_comb = (mod_in.name + '_to_' + mod_out.name + '_epoch_'
                               + str(epoch).zfill(4) + '.png');
                    fn = os.path.join(exp.flags.dir_swapping, fn_comb);
                    swap_plot = plot.create_fig(fn, rec, 11, save_figure=exp.flags.save_figure);
                    swap_plots[mod_in.name + '_' + mod_out.name] = swap_plot;
    return swap_plots;


def generate_conditional_fig_M(exp, epoch, M):
    model = exp.mm_vae;
    mods = exp.modalities;
    samples = exp.test_samples;
    subsets = exp.subsets;

    # get style from random sampling
    random_styles = model.get_random_styles(10);

    cond_plots = dict();
    for k, s_key in enumerate(subsets.keys()):
        subset = subsets[s_key];
        num_mod_s = len(subset);

        if num_mod_s == M:
            s_in = subset;
            for l, m_key_out in enumerate(mods.keys()):
                mod_out = mods[m_key_out];
                rec = torch.zeros(exp.plot_img_size,
                                  dtype=torch.float32).repeat(100 + M*10,1,1,1);
                for m, sample in enumerate(samples):
                    for n, mod_in in enumerate(s_in):
                        c_in = mod_in.plot_data(sample[mod_in.name]);
                        rec[m + n*10, :, :, :] = c_in;
                cond_plots[s_key + '__' + mod_out.name] = rec;

            # style transfer
            for i in range(len(samples)):
                for j in range(len(samples)):
                    i_batch = dict();
                    for o, mod in enumerate(s_in):
                        i_batch[mod.name] = samples[j][mod.name].unsqueeze(0);
                    latents = model.inference(i_batch, num_samples=1)
                    c_in = latents['subsets'][s_key];
                    c_rep = model.reparameterize(mu=c_in[0], logvar=c_in[1]);

                    style = dict();
                    for l, m_key_out in enumerate(mods.keys()):
                        mod_out = mods[m_key_out];
                        if exp.flags.factorized_representation:
                            style[mod_out.name] = random_styles[mod_out.name][i].unsqueeze(0);
                        else:
                            style[mod_out.name] = None;
                    cond_mod_in = {'content': c_rep, 'style': style};
                    cond_gen_samples = model.generate_from_latents(cond_mod_in);

                    for l, m_key_out in enumerate(mods.keys()):
                        mod_out = mods[m_key_out];
                        rec = cond_plots[s_key + '__' + mod_out.name];
                        squeezed = cond_gen_samples[mod_out.name].squeeze(0);
                        p_out = mod_out.plot_data(squeezed);
                        rec[(i+M) * 10 + j, :, :, :] = p_out;
                        cond_plots[s_key + '__' + mod_out.name] = rec;

    for k, s_key_in in enumerate(subsets.keys()):
        subset = subsets[s_key_in];
        if len(subset) == M:
            s_in = subset;
            for l, m_key_out in enumerate(mods.keys()):
                mod_out = mods[m_key_out];
                rec = cond_plots[s_key_in + '__' + mod_out.name];
                fn_comb = (s_key_in + '_to_' + mod_out.name + '_epoch_' +
                           str(epoch).zfill(4) + '.png')
                fn_out = os.path.join(exp.flags.dir_cond_gen, fn_comb);
                plot_out = plot.create_fig(fn_out, rec, 10, save_figure=exp.flags.save_figure);
                cond_plots[s_key_in + '__' + mod_out.name] = plot_out;
    return cond_plots;


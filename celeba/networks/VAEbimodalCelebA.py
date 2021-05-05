import os

import torch
import torch.nn as nn

from utils import utils
from utils.BaseMMVae import BaseMMVae

class VAEbimodalCelebA(BaseMMVae, nn.Module):
    def __init__(self, flags, modalities, subsets):
        super().__init__(flags, modalities, subsets);
        self.flags = flags;
        self.encoder_m1 = modalities['img'].encoder;
        self.encoder_m2 = modalities['text'].encoder;
        self.decoder_m1 = modalities['img'].decoder;
        self.decoder_m2 = modalities['text'].decoder;
        self.lhood_m1 = modalities['img'].likelihood;
        self.lhood_m2 = modalities['text'].likelihood;
        self.encoder_m1 = self.encoder_m1.to(flags.device);
        self.decoder_m1 = self.decoder_m1.to(flags.device);
        self.encoder_m2 = self.encoder_m2.to(flags.device);
        self.decoder_m2 = self.decoder_m2.to(flags.device);


    def forward(self, input_batch):
        latents = self.inference(input_batch);
        results = dict();
        results['latents'] = latents;
        results['group_distr'] = latents['joint'];
        class_embeddings = utils.reparameterize(latents['joint'][0],
                                                latents['joint'][1]);
        div = self.calc_joint_divergence(latents['mus'],
                                         latents['logvars'],
                                         latents['weights']);
        for k, key in enumerate(div.keys()):
            results[key] = div[key];

        results_rec = dict();
        enc_mods = latents['modalities'];
        if 'img' in input_batch.keys():
            m1_s_mu, m1_s_logvar = enc_mods['img_style'];
            if self.flags.factorized_representation:
                m1_s_embeddings = utils.reparameterize(mu=m1_s_mu, logvar=m1_s_logvar);
            else:
                m1_s_embeddings = None;
            m1_rec = self.lhood_m1(*self.decoder_m1(m1_s_embeddings, class_embeddings));
            results_rec['img'] = m1_rec;
        if 'text' in input_batch.keys():
            m2_s_mu, m2_s_logvar = enc_mods['text_style'];
            if self.flags.factorized_representation:
                m2_s_embeddings = utils.reparameterize(mu=m2_s_mu, logvar=m2_s_logvar);
            else:
                m2_s_embeddings = None;
            m2_rec = self.lhood_m2(*self.decoder_m2(m2_s_embeddings, class_embeddings));
            results_rec['text'] = m2_rec;
        results['rec'] = results_rec;
        return results;


    def encode(self, input_batch):
        latents = dict();
        if 'img' in input_batch.keys():
            i_m1 = input_batch['img'];
            latents['img'] = self.encoder_m1(i_m1)
            latents['img_style'] = latents['img'][:2]
            latents['img'] = latents['img'][2:]
        else:
            latents['img_style'] = [None, None];
            latents['img'] = [None, None];

        if 'text' in input_batch.keys():
            i_m2 = input_batch['text'];
            latents['text'] = self.encoder_m2(i_m2)
            latents['text_style'] = latents['text'][:2]
            latents['text'] = latents['text'][2:]
        else:
            latents['text_style'] = [None, None];
            latents['text'] = [None, None];
        return latents;


    def get_random_styles(self, num_samples):
        if self.flags.factorized_representation:
            z_style_1 = torch.randn(num_samples, self.flags.style_img_dim);
            z_style_2 = torch.randn(num_samples, self.flags.style_text_dim);
            z_style_1 = z_style_1.to(self.flags.device);
            z_style_2 = z_style_2.to(self.flags.device);
        else:
            z_style_1 = None;
            z_style_2 = None;
        styles = {'img': z_style_1, 'text': z_style_1};
        return styles;


    def get_random_style_dists(self, num_samples):
        s1_mu = torch.zeros(num_samples,
                            self.flags.style_img_dim).to(self.flags.device)
        s1_logvar = torch.zeros(num_samples, self.flags.style_img_dim).to(self.flags.device);
        s2_mu = torch.zeros(num_samples, self.flags.style_text_dim).to(self.flags.device)
        s2_logvar = torch.zeros(num_samples,
                                self.flags.style_text_dim).to(self.flags.device);
        m1_dist = [s1_mu, s1_logvar];
        m2_dist = [s2_mu, s2_logvar];
        styles = {'img': m1_dist, 'text': m2_dist};
        return styles;


    def generate_sufficient_statistics_from_latents(self, latents):
        content = latents['content']
        suff_stats = dict();
        if 'img' in latents['style'].keys():
            style_m1 = latents['style']['img'];
            cond_gen_m1 = self.lhood_m1(*self.decoder_m1(style_m1, content));
            suff_stats['img'] = cond_gen_m1;
        if 'text' in latents['style'].keys():
            style_m2 = latents['style']['text'];
            cond_gen_m2 = self.lhood_m2(*self.decoder_m2(style_m2, content));
            suff_stats['text'] = cond_gen_m2;
        return suff_stats;


    def save_networks(self):
        torch.save(self.encoder_m1.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m1))
        torch.save(self.decoder_m1.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m1))
        torch.save(self.encoder_m2.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.encoder_save_m2))
        torch.save(self.decoder_m2.state_dict(), os.path.join(self.flags.dir_checkpoints, self.flags.decoder_save_m2))



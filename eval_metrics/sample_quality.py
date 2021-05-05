import os

import numpy as np
import glob 

import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from fid.inception import InceptionV3
from fid.fid_score import get_activations
from fid.fid_score import calculate_frechet_distance

from utils import text as text
import prd_score.prd_score as prd


def calc_inception_features(exp, dims=2048, batch_size=128):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx],
                        path_state_dict=exp.flags.inception_state_dict)
    model = model.to(exp.flags.device);

    paths = exp.paths_fid;
    for m, m_key in enumerate(exp.modalities.keys()):
        mod = exp.modalities[m_key];
        if mod.gen_quality_eval:
            for k, key in enumerate(paths.keys()):
                if key != '':
                    dir_gen = paths[key];
                    if not os.path.exists(dir_gen):
                        raise RuntimeError('Invalid path: %s' % dir_gen)
                    files_gen = glob.glob(os.path.join(dir_gen, mod.name, '*' +
                                                       mod.file_suffix))
                    fn = os.path.join(exp.flags.dir_gen_eval_fid,
                                      key + '_' + mod.name + '_activations.npy');
                    act_gen = get_activations(files_gen, model, batch_size, dims,
                                              True, verbose=False);
                    np.save(fn, act_gen);


def load_inception_activations(exp):
    paths = exp.paths_fid;
    acts = dict();
    for m, m_key in enumerate(exp.modalities.keys()):
        mod = exp.modalities[m_key];
        if mod.gen_quality_eval:
            acts[mod.name] = dict();
            for k, key in enumerate(paths.keys()):
                if key != '':
                    fn = os.path.join(exp.flags.dir_gen_eval_fid,
                                      key + '_' + mod.name + '_activations.npy');
                    feats = np.load(fn);
                    acts[mod.name][key] = feats;
    return acts;


def calculate_inception_features_for_gen_evaluation(flags, paths, modality=None, dims=2048, batch_size=128):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx], path_state_dict=flags.inception_state_dict)
    model = model.to(flags.device);

    if 'random' in list(paths.keys()):
        dir_rand_gen = paths['random'];
        if not os.path.exists(dir_rand_gen):
            raise RuntimeError('Invalid path: %s' % dir_rand_gen)
        if modality is not None:
            files_rand_gen = glob.glob(os.path.join(dir_rand_gen, modality, '*' + '.png'))
            filename_random = os.path.join(flags.dir_gen_eval_fid_random, 'random_sampling_' + modality + '_activations.npy');
        else:
            files_rand_gen = glob.glob(os.path.join(dir_rand_gen, '*.png'));
            filename_random = os.path.join(flags.dir_gen_eval_fid_random, 'random_img_activations.npy')
        act_rand_gen = get_activations(files_rand_gen, model, batch_size, dims,
                                       True, verbose=False);
        np.save(filename_random, act_rand_gen);

    if 'dynamic_prior' in list(paths.keys()):
        dirs_dyn_prior = paths['dynamic_prior'];
        for k, key in enumerate(dirs_dyn_prior.keys()):
            if not os.path.exists(dirs_dyn_prior[key]):
                raise RuntimeError('Invalid path: %s' % dirs_dyn_prior[key])
            files_dyn_gen = glob.glob(os.path.join(dirs_dyn_prior[key], modality, '*' + '.png'))
            filename_dyn = os.path.join(dirs_dyn_prior[key], key + '_' + modality + '_activations.npy')
            act_cond_gen = get_activations(files_dyn_gen, model, batch_size,
                                           dims, True, verbose=False);
            np.save(filename_dyn, act_cond_gen);

    if 'conditional' in list(paths.keys()):
        dir_cond_gen = paths['conditional'];
        if not os.path.exists(dir_cond_gen):
            raise RuntimeError('Invalid path: %s' % dir_cond_gen)
        if modality is not None:
            files_cond_gen = glob.glob(os.path.join(dir_cond_gen, modality, '*' + '.png'))
            filename_conditional = os.path.join(dir_cond_gen, 'cond_gen_' + modality + '_activations.npy')
        else:
            files_cond_gen = glob.glob(os.path.join(dir_cond_gen, '*.png'));
            filename_conditional = os.path.join(flags.dir_gen_eval_fid_cond_gen, 'conditional_img_activations.npy')
        act_cond_gen = get_activations(files_cond_gen, model, batch_size, dims,
                                       True, verbose=False);
        np.save(filename_conditional, act_cond_gen);

    if 'conditional_2a1m' in list(paths.keys()):
        dirs_cond_gen = paths['conditional_2a1m'];
        for k, key in enumerate(dirs_cond_gen.keys()):
            if not os.path.exists(dirs_cond_gen[key]):
                raise RuntimeError('Invalid path: %s' % dirs_cond_gen[key])
            files_cond_gen = glob.glob(os.path.join(dirs_cond_gen[key], modality, '*' + '.png'))
            filename_conditional = os.path.join(dirs_cond_gen[key], key + '_' + modality + '_activations.npy')
            act_cond_gen = get_activations(files_cond_gen, model, batch_size,
                                           dims, True, verbose=False);
            np.save(filename_conditional, act_cond_gen);

    if 'conditional_1a2m' in list(paths.keys()):
        dirs_cond_gen = paths['conditional_1a2m'];
        for k, key in enumerate(dirs_cond_gen.keys()):
            if not os.path.exists(dirs_cond_gen[key]):
                raise RuntimeError('Invalid path: %s' % dirs_cond_gen[key])
            files_cond_gen = glob.glob(os.path.join(dirs_cond_gen[key], modality, '*' + '.png'))
            filename_conditional = os.path.join(dirs_cond_gen[key], key + '_' + modality + '_activations.npy')
            act_cond_gen = get_activations(files_cond_gen, model, batch_size,
                                           dims, True, verbose=False);
            np.save(filename_conditional, act_cond_gen);

    if 'real' in list(paths.keys()):
        dir_real = paths['real'];
        if not os.path.exists(dir_real):
            raise RuntimeError('Invalid path: %s' % dir_real)
        if modality is not None:
            files_real = glob.glob(os.path.join(dir_real, modality, '*' + '.png'));
            filename_real = os.path.join(flags.dir_gen_eval_fid_real, 'real_' + modality + '_activations.npy');
        else:
            files_real = glob.glob(os.path.join(dir_real, '*.png'));
            filename_real = os.path.join(flags.dir_gen_eval_fid_real, 'real_img_activations.npy')
        act_real = get_activations(files_real, model, batch_size, dims, True, verbose=False);
        np.save(filename_real, act_real);



def calculate_fid(feats_real, feats_gen):
    mu_real = np.mean(feats_real, axis=0)
    sigma_real = np.cov(feats_real, rowvar=False)
    mu_gen = np.mean(feats_gen, axis=0)
    sigma_gen = np.cov(feats_gen, rowvar=False)
    fid = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)
    return fid;


def calculate_fid_dict(feats_real, dict_feats_gen):
    dict_fid = dict();
    for k, key in enumerate(dict_feats_gen.keys()):
        feats_gen = dict_feats_gen[key];
        dict_fid[key] = calculate_fid(feats_real, feats_gen);
    return dict_fid;


def calculate_prd(feats_real, feats_gen):
    prd_val = prd.compute_prd_from_embedding(feats_real, feats_gen)
    ap = np.mean(prd_val);
    return ap;


def calculate_prd_dict(feats_real, dict_feats_gen):
    dict_fid = dict();
    for k, key in enumerate(dict_feats_gen.keys()):
        feats_gen = dict_feats_gen[key];
        dict_fid[key] = calculate_prd(feats_real, feats_gen);
    return dict_fid;


def get_clf_activations(flags, data, model):
    model.eval();
    act = model.get_activations(data);
    act = act.cpu().data.numpy().reshape(flags.batch_size, -1)
    return act;


def calc_prd_score(exp):
    calc_inception_features(exp);
    acts = load_inception_activations(exp);
    ap_prds = dict();
    for m, m_key in enumerate(exp.modalities.keys()):
        mod = exp.modalities[m_key];
        if mod.gen_quality_eval:
            for k, key in enumerate(exp.subsets):
                if key == '':
                    continue;
                ap_prd = calculate_prd(acts[mod.name]['real'],
                                       acts[mod.name][key]);
                ap_prds[key + '_' + mod.name] = ap_prd;

    for m, m_key in enumerate(exp.modalities.keys()):
        mod = exp.modalities[m_key];
        if mod.gen_quality_eval:
            ap_prd = calculate_prd(acts[mod.name]['real'],
                                   acts[mod.name]['random']);
            ap_prds['random_' + mod.name] = ap_prd;
    return ap_prds;









import numpy as np
import math
import torch

from utils import utils
from divergence_measures.mm_div import alpha_poe

LOG2PI = float(np.log(2.0 * math.pi))


def get_latent_samples(model, latents, n_imp_samples, mod_names=None):
    l_c = latents['content'];
    l_s = latents['style'];
    l_c_m_rep = l_c[0].unsqueeze(0).repeat(n_imp_samples, 1, 1);
    l_c_lv_rep = l_c[1].unsqueeze(0).repeat(n_imp_samples, 1, 1);
    c_emb = model.reparameterize(l_c_m_rep, l_c_lv_rep);
    styles = dict();
    c = {'mu': l_c_m_rep, 'logvar': l_c_lv_rep, 'z': c_emb}
    if model.flags.factorized_representation:
        for k, key in enumerate(l_s.keys()):
            l_s_mod = l_s[key];
            l_s_m_rep = l_s_mod[0].unsqueeze(0).repeat(n_imp_samples, 1, 1);
            l_s_lv_rep = l_s_mod[1].unsqueeze(0).repeat(n_imp_samples, 1, 1);
            s_emb = model.reparameterize(l_s_m_rep, l_s_lv_rep);
            s = {'mu': l_s_m_rep, 'logvar': l_s_lv_rep, 'z': s_emb}
            styles[key] = s;
    else:
        for k, key in enumerate(mod_names):
            styles[key] = None;
    emb = {'content': c, 'style': styles}
    return emb;


def get_dyn_prior(weights, mus, logvars):
    mu_poe, logvar_poe = alpha_poe(weights, mus, logvars);
    return [mu_poe, logvar_poe];


def log_mean_exp(x, dim=1):
    """
    log(1/k * sum(exp(x))): this normalizes x.
    @param x: PyTorch.Tensor
              samples from gaussian
    @param dim: integer (default: 1)
                which dimension to take the mean over
    @return: PyTorch.Tensor
             mean of x
    """
    m = torch.max(x, dim=dim, keepdim=True)[0]
    return m + torch.log(torch.mean(torch.exp(x - m),
                         dim=dim, keepdim=True))


def gaussian_log_pdf(x, mu, logvar):
    """
    Log-likelihood of data given ~N(mu, exp(logvar))
    @param x: samples from gaussian
    @param mu: mean of distribution
    @param logvar: log variance of distribution
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    global LOG2PI
    log_pdf = -0.5 * LOG2PI - logvar / 2. - torch.pow(x - mu, 2) / (2. * torch.exp(logvar))
    return torch.sum(log_pdf, dim=1)


def unit_gaussian_log_pdf(x):
    """
    Log-likelihood of data given ~N(0, 1)
    @param x: PyTorch.Tensor
              samples from gaussian
    @return log_pdf: PyTorch.Tensor
                     log-likelihood
    """
    global LOG2PI
    log_pdf = -0.5 * LOG2PI - math.log(1.) / 2. - torch.pow(x, 2) / 2.
    return torch.sum(log_pdf, dim=1)


def log_marginal_estimate(flags, n_samples, likelihood, image, style, content, dynamic_prior=None):
    r"""Estimate log p(x). NOTE: this is not the objective that
    should be directly optimized.
    @param ss_list: list of sufficient stats, i.e., list of
                        torch.Tensor (batch size x # samples x 784)
    @param image: torch.Tensor (batch size x 784)
                  original observed image
    @param z: torch.Tensor (batch_size x # samples x z dim)
              samples drawn from variational distribution
    @param mu: torch.Tensor (batch_size x # samples x z dim)
               means of variational distribution
    @param logvar: torch.Tensor (batch_size x # samples x z dim)
                   log-variance of variational distribution
    """
    batch_size = flags.batch_size;
    if style is not None:
        z_style = style['z'];
        logvar_style = style['logvar'];
        mu_style = style['mu'];
        n, z_style_dim = z_style.size()
        style_log_q_z_given_x_2d = gaussian_log_pdf(z_style, mu_style, logvar_style);
        log_p_z_2d_style = unit_gaussian_log_pdf(z_style)

    d_shape = image.shape;
    if len(d_shape) == 3:
        image = image.unsqueeze(0).repeat(n_samples, 1, 1, 1);
        image = image.view(batch_size*n_samples, d_shape[-2], d_shape[-1])
    elif len(d_shape) == 4:
        image = image.unsqueeze(0).repeat(n_samples, 1, 1, 1, 1);
        image = image.view(batch_size*n_samples, d_shape[-3], d_shape[-2],
                           d_shape[-1])
    
    z_content = content['z']
    mu_content = content['mu'];
    logvar_content = content['logvar'];
    log_p_x_given_z_2d = likelihood.log_prob(image).view(batch_size*n_samples,
                                                        -1).sum(dim=1)
    content_log_q_z_given_x_2d = gaussian_log_pdf(z_content, mu_content, logvar_content);


    if dynamic_prior is None:
        log_p_z_2d_content = unit_gaussian_log_pdf(z_content)
    else:
        mu_prior = dynamic_prior['mu'];
        logvar_prior = dynamic_prior['logvar'];
        log_p_z_2d_content = gaussian_log_pdf(z_content, mu_prior, logvar_prior);

    if style is not None:
        log_p_z_2d = log_p_z_2d_style+log_p_z_2d_content;
        log_q_z_given_x_2d = style_log_q_z_given_x_2d + content_log_q_z_given_x_2d
    else:
        log_p_z_2d = log_p_z_2d_content;
        log_q_z_given_x_2d = content_log_q_z_given_x_2d;
    log_weight_2d = log_p_x_given_z_2d + log_p_z_2d - log_q_z_given_x_2d;
    log_weight = log_weight_2d.view(batch_size, n_samples)

    # need to compute normalization constant for weights
    # i.e. log ( mean ( exp ( log_weights ) ) )
    log_p = log_mean_exp(log_weight, dim=1)
    return torch.mean(log_p)


def log_joint_estimate(flags, n_samples, likelihoods, targets, styles, content, dynamic_prior=None):
    r"""Estimate log p(x,y).
    @param recon_image: torch.Tensor (batch size x # samples x 784)
                        reconstructed means on bernoulli
    @param image: torch.Tensor (batch size x 784)
                  original observed image
    @param recon_label: torch.Tensor (batch_size x # samples x n_class)
                        reconstructed logits
    @param label: torch.Tensor (batch_size)
                  original observed labels
    @param z: torch.Tensor (batch_size x # samples x z dim)
              samples drawn from variational distribution
    @param mu: torch.Tensor (batch_size x # samples x z dim)
               means of variational distribution
    @param logvar: torch.Tensor (batch_size x # samples x z dim)
                   log-variance of variational distribution
    """
    batch_size = flags.batch_size;
    if styles is not None:
        styles_log_q_z_given_x_2d = dict();
        styles_p_z_2d = dict();
        for key in styles.keys():
            if styles[key] is not None:
                style_m = styles[key];
                z_style_m = style_m['z'];
                logvar_style_m = style_m['logvar'];
                mu_style_m = style_m['mu'];
                style_m_log_q_z_given_x_2d = gaussian_log_pdf(z_style_m, mu_style_m, logvar_style_m);
                log_p_z_2d_style_m = unit_gaussian_log_pdf(z_style_m)
                styles_log_q_z_given_x_2d[key] = style_m_log_q_z_given_x_2d;
                styles_p_z_2d[key] = log_p_z_2d_style_m;

    z_content = content['z']
    mu_content = content['mu'];
    logvar_content = content['logvar'];

    num_mods = len(styles.keys())
    log_px_zs = torch.zeros(num_mods, batch_size * n_samples);
    log_px_zs = log_px_zs.to(flags.device);
    for k, key in enumerate(styles.keys()):
        batch_d = targets[key]
        d_shape = batch_d.shape;
        if len(d_shape) == 3:
            batch_d = batch_d.unsqueeze(0).repeat(n_samples, 1, 1, 1);
            batch_d = batch_d.view(batch_size*n_samples, d_shape[-2], d_shape[-1])
        elif len(d_shape) == 4:
            batch_d = batch_d.unsqueeze(0).repeat(n_samples, 1, 1, 1, 1);
            batch_d = batch_d.view(batch_size*n_samples, d_shape[-3], d_shape[-2],
                               d_shape[-1])
        lhood = likelihoods[key]
        log_p_x_given_z_2d = lhood.log_prob(batch_d).view(batch_size * n_samples, -1).sum(dim=1);
        log_px_zs[k] = log_p_x_given_z_2d;

    # compute components of likelihood estimate
    log_joint_zs_2d = log_px_zs.sum(0)  # sum over modalities

    if dynamic_prior is None:
        log_p_z_2d_content = unit_gaussian_log_pdf(z_content)
    else:
        mu_prior = dynamic_prior['mu'];
        logvar_prior = dynamic_prior['logvar'];
        log_p_z_2d_content = gaussian_log_pdf(z_content, mu_prior, logvar_prior);

    content_log_q_z_given_x_2d = gaussian_log_pdf(z_content, mu_content, logvar_content);
    log_p_z_2d = log_p_z_2d_content;
    log_q_z_given_x_2d = content_log_q_z_given_x_2d;
    if styles is not None:
        for k, key in enumerate(styles.keys()):
            if key in styles_p_z_2d and key in styles_log_q_z_given_x_2d:
                log_p_z_2d += styles_p_z_2d[key];
                log_q_z_given_x_2d += styles_log_q_z_given_x_2d[key];

    log_weight_2d = log_joint_zs_2d + log_p_z_2d - log_q_z_given_x_2d;
    log_weight = log_weight_2d.view(batch_size, n_samples)
    log_p = log_mean_exp(log_weight, dim=1)
    return torch.mean(log_p)



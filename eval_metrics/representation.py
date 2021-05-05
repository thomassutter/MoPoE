import sys
import os

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def train_clf_lr_all_subsets(exp):
    mm_vae = exp.mm_vae;
    mm_vae.eval();
    subsets = exp.subsets;

    d_loader = DataLoader(exp.dataset_train, batch_size=exp.flags.batch_size,
                        shuffle=True,
                        num_workers=8, drop_last=True);

    bs = exp.flags.batch_size;
    num_batches_epoch = int(exp.dataset_train.__len__() /float(bs));
    class_dim = exp.flags.class_dim;
    n_samples = int(exp.dataset_train.__len__());
    data_train = dict();
    for k, s_key in enumerate(subsets.keys()):
        if s_key != '':
            data_train[s_key] = np.zeros((n_samples,
                                          class_dim))
    all_labels = np.zeros((n_samples, len(exp.labels)))
    for it, batch in enumerate(d_loader):
        batch_d = batch[0];
        batch_l = batch[1];
        for k, m_key in enumerate(batch_d.keys()):
            batch_d[m_key] = batch_d[m_key].to(exp.flags.device);
        inferred = mm_vae.inference(batch_d);
        lr_subsets = inferred['subsets'];
        all_labels[(it*bs):((it+1)*bs), :] = np.reshape(batch_l, (bs,
                                                                  len(exp.labels)));
        for k, key in enumerate(lr_subsets.keys()):
            data_train[key][(it*bs):((it+1)*bs), :] = lr_subsets[key][0].cpu().data.numpy();

    n_train_samples = exp.flags.num_training_samples_lr;
    rand_ind_train = np.random.randint(n_samples, size=n_train_samples)
    labels = all_labels[rand_ind_train,:]
    for k, s_key in enumerate(subsets.keys()):
        if s_key != '':
            d = data_train[s_key];
            data_train[s_key] = d[rand_ind_train, :]
    clf_lr = train_clf_lr(exp, data_train, labels);
    return clf_lr;
 

def test_clf_lr_all_subsets(epoch, clf_lr, exp):
    mm_vae = exp.mm_vae;
    mm_vae.eval();
    subsets = exp.subsets;

    lr_eval = dict();
    for l, label_str in enumerate(exp.labels):
        lr_eval[label_str] = dict();
        for k, s_key in enumerate(subsets.keys()):
            if s_key != '':
                lr_eval[label_str][s_key] = [];

    d_loader = DataLoader(exp.dataset_test, batch_size=exp.flags.batch_size,
                        shuffle=True,
                        num_workers=8, drop_last=True);

    num_batches_epoch = int(exp.dataset_test.__len__() /float(exp.flags.batch_size));
    for iteration, batch in enumerate(d_loader):
        batch_d = batch[0];
        batch_l = batch[1];
        for k, m_key in enumerate(batch_d.keys()):
            batch_d[m_key] = batch_d[m_key].to(exp.flags.device);
        inferred = mm_vae.inference(batch_d);
        lr_subsets = inferred['subsets'];
        data_test = dict();
        for k, key in enumerate(lr_subsets.keys()):
            data_test[key] = lr_subsets[key][0].cpu().data.numpy();
        evals = classify_latent_representations(exp,
                                                epoch,
                                                clf_lr,
                                                data_test,
                                                batch_l);
        for l, label_str in enumerate(exp.labels):
            eval_label = evals[label_str];
            for k, s_key in enumerate(eval_label.keys()):
                lr_eval[label_str][s_key].append(eval_label[s_key]);
    for l, l_key in enumerate(lr_eval.keys()):
        lr_eval_label = lr_eval[l_key];
        for k, s_key in enumerate(lr_eval_label.keys()):
            lr_eval[l_key][s_key] = exp.mean_eval_metric(lr_eval_label[s_key]);
    return lr_eval;


def classify_latent_representations(exp, epoch, clf_lr, data, labels):
    labels = np.array(np.reshape(labels, (labels.shape[0], len(exp.labels))));
    eval_all_labels = dict()
    for l, label_str in enumerate(exp.labels):
        gt = labels[:, l];
        clf_lr_label = clf_lr[label_str];
        eval_all_reps = dict();
        for s_key in data.keys():
            data_rep = data[s_key];
            clf_lr_rep = clf_lr_label[s_key];
            y_pred_rep = clf_lr_rep.predict(data_rep);
            eval_label_rep = exp.eval_metric(gt.ravel(),
                                             y_pred_rep.ravel());
            eval_all_reps[s_key] = eval_label_rep;
        eval_all_labels[label_str] = eval_all_reps;
    return eval_all_labels;


def train_clf_lr(exp, data, labels):
    labels = np.reshape(labels, (labels.shape[0], len(exp.labels)));
    clf_lr_labels = dict();
    for l, label_str in enumerate(exp.labels):
        gt = labels[:, l];
        clf_lr_reps = dict();
        for s_key in data.keys():
            data_rep = data[s_key];
            clf_lr_s = LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto', max_iter=1000);
            clf_lr_s.fit(data_rep, gt.ravel());
            clf_lr_reps[s_key] = clf_lr_s;
        clf_lr_labels[label_str] = clf_lr_reps;
    return clf_lr_labels;






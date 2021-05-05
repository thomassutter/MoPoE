

import argparse

parser = argparse.ArgumentParser()

# DATASET NAME
# to be specified by experiments themselves
# parser.add_argument('--dataset', type=str, default='SVHN_MNIST_text', help="name of the dataset")

# TRAINING
parser.add_argument('--batch_size', type=int, default=256, help="batch size for training")
parser.add_argument('--initial_learning_rate', type=float, default=0.001, help="starting learning rate")
parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")
parser.add_argument('--start_epoch', type=int, default=0, help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=100, help="flag to indicate the final epoch of training")

# DATA DEPENDENT
parser.add_argument('--class_dim', type=int, default=20, help="dimension of common factor latent space")
# to be set by experiments themselves
# parser.add_argument('--style_m1_dim', type=int, default=0, help="dimension of varying factor latent space")
# parser.add_argument('--style_m2_dim', type=int, default=0, help="dimension of varying factor latent space")
# parser.add_argument('--style_m3_dim', type=int, default=0, help="dimension of varying factor latent space")
# parser.add_argument('--len_sequence', type=int, default=8, help="length of sequence")
# parser.add_argument('--dim', type=int, default=64, help="number of classes on which the data set trained")
# parser.add_argument('--data_multiplications', type=int, default=20, help="number of pairs per sample")
# parser.add_argument('--num_hidden_layers', type=int, default=1, help="number of channels in images")
# parser.add_argument('--likelihood_m1', type=str, default='laplace', help="output distribution")
# parser.add_argument('--likelihood_m2', type=str, default='laplace', help="output distribution")
# parser.add_argument('--likelihood_m3', type=str, default='categorical', help="output distribution")

# SAVE and LOAD
parser.add_argument('--mm_vae_save', type=str, default='mm_vae', help="model save for vae_bimodal")
parser.add_argument('--load_saved', type=bool, default=False, help="flag to indicate if a saved model will be loaded")
# to bet set by experiments themselves
# parser.add_argument('--encoder_save_m1', type=str, default='encoderM1', help="model save for encoder")
# parser.add_argument('--encoder_save_m2', type=str, default='encoderM2', help="model save for encoder")
# parser.add_argument('--encoder_save_m3', type=str, default='encoderM3', help="model save for decoder")
# parser.add_argument('--decoder_save_m1', type=str, default='decoderM1', help="model save for decoder")
# parser.add_argument('--decoder_save_m2', type=str, default='decoderM2', help="model save for decoder")
# parser.add_argument('--decoder_save_m3', type=str, default='decoderM3', help="model save for decoder")
# parser.add_argument('--clf_save_m1', type=str, default='clf_m1', help="model save for clf")
# parser.add_argument('--clf_save_m2', type=str, default='clf_m2', help="model save for clf")
# parser.add_argument('--clf_save_m3', type=str, default='clf_m3', help="model save for clf")

# DIRECTORIES
# clfs
parser.add_argument('--dir_clf', type=str, default='../clf', help="directory where clf is stored")
# data
parser.add_argument('--dir_data', type=str, default='../data', help="directory where data is stored")
# experiments
parser.add_argument('--dir_experiment', type=str, default='/tmp/multilevel_multimodal_vae_swapping', help="directory to save generated samples in")
# fid
parser.add_argument('--dir_fid', type=str, default=None, help="directory to save generated samples for fid score calculation")
#fid_score
parser.add_argument('--inception_state_dict', type=str, default='../inception_state_dict.pth', help="path to inception v3 state dict")

# EVALUATION
parser.add_argument('--use_clf', default=False, action="store_true",
                    help="flag to indicate if generates samples should be classified")
parser.add_argument('--calc_nll', default=False, action="store_true",
                    help="flag to indicate calculation of nll")
parser.add_argument('--eval_lr', default=False, action="store_true",
                    help="flag to indicate evaluation of lr")
parser.add_argument('--calc_prd', default=False, action="store_true",
                    help="flag to indicate calculation of prec-rec for gen model")
parser.add_argument('--save_figure', default=False, action="store_true",
                    help="flag to indicate if figures should be saved to disk (in addition to tensorboard logs)")
parser.add_argument('--eval_freq', type=int, default=10, help="frequency of evaluation of latent representation of generative performance (in number of epochs)")
parser.add_argument('--eval_freq_fid', type=int, default=10, help="frequency of evaluation of latent representation of generative performance (in number of epochs)")
parser.add_argument('--num_samples_fid', type=int, default=10000,
                    help="number of samples the calculation of fid is based on")
parser.add_argument('--num_training_samples_lr', type=int, default=500,
                    help="number of training samples to train the lr clf")

#multimodal
parser.add_argument('--method', type=str, default='poe', help='choose method for training the model')
parser.add_argument('--modality_jsd', type=bool, default=False, help="modality_jsd")
parser.add_argument('--modality_poe', type=bool, default=False, help="modality_poe")
parser.add_argument('--modality_moe', type=bool, default=False, help="modality_moe")
parser.add_argument('--joint_elbo', type=bool, default=False, help="modality_moe")
parser.add_argument('--poe_unimodal_elbos', type=bool, default=True, help="unimodal_klds")
parser.add_argument('--factorized_representation', action='store_true', default=False, help="factorized_representation")

# LOSS TERM WEIGHTS
parser.add_argument('--beta', type=float, default=5.0, help="default weight of sum of weighted divergence terms")
parser.add_argument('--beta_style', type=float, default=1.0, help="default weight of sum of weighted style divergence terms")
parser.add_argument('--beta_content', type=float, default=1.0, help="default weight of sum of weighted content divergence terms")
# to be specified by experiments themselves
# parser.add_argument('--beta_m1_style', type=float, default=1.0, help="default weight divergence term style modality 1")
# parser.add_argument('--beta_m2_style', type=float, default=1.0, help="default weight divergence term style modality 2")
# parser.add_argument('--beta_m3_style', type=float, default=1.0, help="default weight divergence term style modality 2")
# parser.add_argument('--div_weight_m1_content', type=float, default=0.25, help="default weight divergence term content modality 1")
# parser.add_argument('--div_weight_m2_content', type=float, default=0.25, help="default weight divergence term content modality 2")
# parser.add_argument('--div_weight_m3_content', type=float, default=0.25, help="default weight divergence term content modality 2")
# parser.add_argument('--div_weight_uniform_content', type=float, default=0.25, help="default weight divergence term prior")




from utils.BaseFlags import parser as parser

parser.add_argument('--dataset', type=str, default='MMNIST', help="name of the dataset")

parser.add_argument('--style_dim', type=int, default=0, help="style dimensionality")  # TODO: use modality-specific style dimensions?
parser.add_argument('--num_classes', type=int, default=10, help="number of classes on which the data set trained")
parser.add_argument('--len_sequence', type=int, default=8, help="length of sequence")
parser.add_argument('--img_size_m1', type=int, default=28, help="img dimension (width/height)")
parser.add_argument('--num_channels_m1', type=int, default=1, help="number of channels in images")
parser.add_argument('--img_size_m2', type=int, default=32, help="img dimension (width/height)")
parser.add_argument('--num_channels_m2', type=int, default=3, help="number of channels in images")
parser.add_argument('--dim', type=int, default=64, help="number of classes on which the data set trained")
parser.add_argument('--data_multiplications', type=int, default=1, help="number of pairs per sample")
parser.add_argument('--num_hidden_layers', type=int, default=1, help="number of channels in images")
parser.add_argument('--likelihood', type=str, default='laplace', help="output distribution")

# data
parser.add_argument('--unimodal-datapaths-train', nargs="+", type=str, help="directories where training data is stored")
parser.add_argument('--unimodal-datapaths-test', nargs="+", type=str, help="directories where test data is stored")
parser.add_argument('--pretrained-classifier-paths', nargs="+", type=str, help="paths to pretrained classifiers")

# multimodal
parser.add_argument('--subsampled_reconstruction', default=True, help="subsample reconstruction path")
parser.add_argument('--include_prior_expert', action='store_true', default=False, help="factorized_representation")

# weighting of loss terms
parser.add_argument('--div_weight', type=float, default=None, help="default weight divergence per modality, if None use 1/(num_mods+1).")
parser.add_argument('--div_weight_uniform_content', type=float, default=None, help="default weight divergence term prior, if None use (1/num_mods+1)")

# annealing
parser.add_argument('--kl_annealing', type=int, default=0, help="number of kl annealing steps; 0 if no annealing should be done")

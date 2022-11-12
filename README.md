# MoPoE-VAE
This is the official code for the ICLR 2021 paper "Generalized Multimodal ELBO".
Here is the link to the OpenReview-Site: <https://openreview.net/forum?id=5Y21V0RDBV>

If you have any questions about the code or the paper, we are happy to help!

## Preliminaries

This code was developed and tested with:
- Python version 3.5.6
- PyTorch version 1.4.0
- CUDA version 11.0
- The conda environment defined in `environment.yml`

First, set up the conda enviroment as follows:
```bash
conda env create -f environment.yml  # create conda env
conda activate mopoe                 # activate conda env
```

Second, download the data, inception network, and pretrained classifiers:
```bash
curl -L -o tmp.zip https://drive.google.com/drive/folders/1lr-laYwjDq3AzalaIe9jN4shpt1wBsYM?usp=sharing
unzip tmp.zip
unzip celeba_data.zip -d data/
unzip data_mnistsvhntext.zip -d data/
unzip PolyMNIST.zip -d data/
```

## Experiments

Experiments can be started by running the respective `job_*` script.
To choose between running the MVAE, MMVAE, and MoPoE-VAE, one needs to
change the script's `METHOD` variabe to "poe", "moe", or "joint\_elbo"
respectively.  By default, each experiment uses `METHOD="joint_elbo"`.

### running MNIST-SVHN-Text
```bash
./job_mnistsvhntext
```

### running PolyMNIST
```bash
./job_polymnist
```

### running Bimodal Celeba
```bash
./job_celeba
```

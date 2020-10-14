# Adversially Robust Sparse Models

This toolbox provides implementation for all experiments in the paper Adversarial Robustness of Supervised Sparse Coding, by J. Sulam, R. Muthukumar and R. Arora.

Experiments are separated by jupyter notebooks for clarity. 

### Dependencies:
* numpy, 
* torch
* sklearn
* matplotlib
* auxiliary libs (pdb, time, importlib)
and importantly:
* art: [the Advesarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

### Functionality
Synthetic models and functions are gathered in `adverarial_sparse_toolbox.py`, whereas unsupervised and supervised dictionary models and learning functions are in `sparse_learnable_dictionary.py`.

Comments and suggestions welcome at `jsulam1@jhu.edu`
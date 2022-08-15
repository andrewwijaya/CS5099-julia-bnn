# CS5099 A Probabilistic Deep Learning Implementation In Julia

This project aims to explore Bayesian approaches to deep learning by implementing Variational Inference, and also a simple HMC based sampler. The VI implementation is largely based on the Autograd Bayesian Neural Networks example which is implemented in Python. The HMC implementation is based on the example from AdvancedHMC which uses the No-U-Turn Samples (NUTS). This project compares and evaluates sampling and VI based approaches.

This codebase contains a number of notebooks for the experiments, and some Julia modules which contain the VI implementation. Details of each artefact are described in the final report's 'Artefact' section.

## Requirements
To run the Julia implementations in this codebase you need Julia version 1.7, and Jupyter notebook. There are several other packages that will require installing using the Pkg package. This is all detailed in the final report.

## Variational Inference

To see the results of experiments for BNN using VI check the vi_experiments notebooks.

## Hamiltonian Monte Carlo

The results of experiments using HMC are in the hmc_experiments.

# A Probabilistic Deep Learning Implementation In Julia

This project aims to explore Bayesian approaches to deep learning by implementing Variational Inference, and also a simple HMC based sampler. The VI implementation is largely based on the Autograd Bayesian Neural Networks example which is implemented in Python. The HMC implementation is based on the example from AdvancedHMC which uses the No-U-Turn Samples (NUTS). This project compares and evaluates sampling and VI based approaches.

This codebase contains a number of notebooks for the experiments, and some Julia modules which contain the VI implementation. Details of each artefact are described in the final report's 'Artefact' section.

## Requirements
To run the Julia implementations in this codebase you need Julia version 1.7, and Jupyter notebook. There are several other packages that will require installing using the Pkg package. This is all detailed in the final report.

## Variational Inference

To see the results of experiments for BNN using VI check the vi_experiments notebooks.

## Hamiltonian Monte Carlo

The results of experiments using HMC are in the hmc_experiments.

## Execution Steps

1. Download Julia from the official site: https://julialang.org/downloads/
2. Install Julia
3. Download anaconda distribution from: https://www.anaconda.com/products/distribution
4. For Linux, run
```
jupyter notebook
```
5. For Windows, open an anaconda prompt then run
```
jupyter notebook
```
6. Open Julia prompt and execute: (you should be able to just copy the whole list into prompt and execute)
```
using Pkg
Pkg.add("IJulia")
Pkg.add("Plots")
Pkg.add("Distributions")
Pkg.add("Flux")
Pkg.add("OrderedCollections")
Pkg.add("OMEinsum")
Pkg.add("BenchmarkTools")
Pkg.add(Pkg.PackageSpec(;name="AdvancedHMC", version="0.3.5"))
Pkg.add("ForwardDiff")
Pkg.add("Zygote")
```
7. Unzip the submitted codebase
8. On Jupyter Notebook in your browser, navigate to where the code was extracted from step 7
9. Execute all five notebooks

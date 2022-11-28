# Wavelet-Neural-Operator-for-pdes
Wavelet  Neural  Operator  for  solving  parametric  partialdifferential  equations  in  computational  mechanics  problems

# Learning-physics-from-output-only-data
This repository contains the python codes of the paper 
  > + "A sparse Bayesian framework for discovering interpretable nonlinear stochastic dynamical systems with Gaussian white noise", authored by Tapas Tripura and Souvik Chakraborty. [Article](https://doi.org/10.1016/j.ymssp.2022.109939)
  > + In arXiv version this article can be searched as "Learning governing physics from output only measurements". The arXiv version can be accessed [here](https://arxiv.org/pdf/2208.05609.pdf).

# Files
The main codes, described below, are standalone codes. They can be directly run. However, for the figures in the published article, one needs to run the files in the folder `Paper_figures`. A short despcription on the files are provided below for ease of readers.
  + `Example_1_BScholes.py` is the code to discover physics for the Example 1: Black-Scholes equation [article](https://arxiv.org/pdf/2208.05609.pdf).
  + `Example_2_Duffing.py` is the code to discover physics for the Example 2: Parametrically excited Duffing oscillator [article](https://arxiv.org/pdf/2208.05609.pdf).
  + `Example_3_2DOF.py` is the code to discover physics for the Example 3: 2DOF system [article](https://arxiv.org/pdf/2208.05609.pdf).
  + `Example_4_boucwen.py` is the code to discover physics for the Example 4: Bouc-Wen, with partially observed state variables [article](https://arxiv.org/pdf/2208.05609.pdf).
  - `utils_gibbs.py` is a part of gibbs sampling in section 2.2 [article](https://arxiv.org/pdf/2208.05609.pdf).
  * `utils_library.py` contains useful functions, like, library construction, data-normalization.
  + `utils_response.py` is the code to generate data using stochastic calculus.
The codes for the *Stochastic SINDy* are provided in the folder *Stochastic_Sindy*.

# Dataset
The saved dataset for reproducing the figures in the above article can be accessed at,
> [Saved Datasets](https://drive.google.com/drive/folders/1o5ZoWFjuJwuktp-Kgl9acQUlZ5ALEtZB?usp=sharing)

# BibTex
If you use any part our codes, please cite us at,
```
@article{tripura2023sparse,
  title={A sparse Bayesian framework for discovering interpretable nonlinear stochastic dynamical systems with Gaussian white noise},
  author={Tripura, Tapas and Chakraborty, Souvik},
  journal={Mechanical Systems and Signal Processing},
  volume={187},
  pages={109939},
  year={2023},
  publisher={Elsevier}
}
```

# APA-style
```
Tripura, T., & Chakraborty, S. (2023). A sparse Bayesian framework for discovering interpretable nonlinear stochastic dynamical systems with Gaussian white noise. Mechanical Systems and Signal Processing, 187, 109939.
```

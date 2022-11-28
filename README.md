# Wavelet-Neural-Operator-for-pdes
This repository contains the python codes of the paper 
  > + "Wavelet  Neural  Operator  for  solving  parametric  partialdifferential  equations  in  computational  mechanics  problems", authored by Tapas Tripura and Souvik Chakraborty.
  > + In arXiv version this article can be searched as "Wavelet neural operator: a neural operator for parametric partial differential equations". The arXiv version can be accessed [here](https://arxiv.org/abs/2208.05609).

> Architecture of the wavelet neural operator (WNO). (a) Schematic of the proposed neural operator. (b) A simple WNO with one wavelet kernel integral layer. 
![WNO](WNN.png)

> Construction of the parametric space using multiwavelet decomposition.
![Construction of parameterization space in WNO](WNN_parameter.png)

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
@article{tripura2022wavelet,
  title={Wavelet neural operator: a neural operator for parametric partial differential equations},
  author={Tripura, Tapas and Chakraborty, Souvik},
  journal={arXiv preprint arXiv:2205.02191},
  year={2022}
}
```

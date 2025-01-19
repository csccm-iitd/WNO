# Wavelet-Neural-Operator (WNO)
This repository contains the python codes of the paper 
  > + Tripura, T., & Chakraborty, S. (2023). Wavelet Neural Operator for solving parametric partial differential equations in computational mechanics problems. Computer Methods in Applied Mechanics and Engineering, 404, 115783. [Article](https://doi.org/10.1016/j.cma.2022.115783)
  > + ArXiv version- "Wavelet neural operator: a neural operator for parametric partial differential equations". The arXiv version can be accessed [here](https://arxiv.org/abs/2205.02191).

## New in version 2.0.0
```
 > Added superresolution attribute to the WNO.
 > Added 3D support to the WNO.
 > Improved the interface and readability of the codes.
```

## Architecture of the wavelet neural operator (WNO). 
(a) Schematic of the proposed neural operator. (b) A simple WNO with one wavelet kernel integral layer. 
![WNO](/Github_page_images/WNN.png)

## Construction of the parametric space using multi-level wavelet decomposition.
![Construction of parameterization space in WNO](/Github_page_images/WNN_parameter.png)

## Super resolution using Wavelet Neural Operator.
  > Super resolution in Burgers' diffusion dynamics:
  ![Train at resolution-1024 and Test at resolution-2048](/Github_page_images/Burgers_prediction.png)
  > Super resolution in Navier-Stokes equation with 10000 Reynolds number:
  ![Train in Low resolution](/Github_page_images/Animation_ns_64_3d_1e-4.gif)
  ![Test in High resolution](/Github_page_images/Animation_ns_256_3d_1e-4.gif)

## Files
A short despcription on the files are provided below for ease of readers.
```
  + `wno1d_advection_III.py`: For 1D wave advection equation (time-independent problem).
  + `wno1d_Advection_time_III.py`: For 1D wave advection equation with time-marching (time-dependent problem).
  + `wno1d_Burger_discontinuous.py`: For 1D Burgers' equation with discontinuous field (time-dependent problem).
  + `wno1d_Burgers.py`: For 1D Burger's equation (time-independent problem).
  + `wno2d_AC_dwt.py`: For 2D Allen-Cahn reaction-diffusion equation (time-independent problem).
  + `wno2d_Darcy_notch_cwt.py`: For 2D Darcy equation using Slim Continuous Wavelet Transform (time-independent problem).
  + `wno2d_Darcy_notch_dwt.py`: For 2D Darcy equation using Discrete wavelet transform (time-independent problem).
  + `wno2d_NS_cwt.py`: For 2D Navier-Stokes equation using Slim Continuous Wavelet Transform (time-dependent problem).
  + `wno2d_NS_dwt.py`: For 2D Navier-Stokes equation using Discrete Wavelet Transform (time-dependent problem).
  + `wno2d_Temperature_Daily_Avg.py`: For forecasting daily averaged 2m air temperature (time-dependent problem).
  + `wno2d_Temperature_Monthly_Avg.py`: For forecasting monthly averaged 2m air temperature (time-independent problem).
  + `wno3d_NS.py`: For 2D Navier-Stokes equation using 3D WNO (as a time-independent problem).

  + `Test_wno_super_1d_Burgers.py`: An example of Testing on new data with supersolution.
  
  + `utils.py` contains some useful functions for data handling (improvised from [FNO paper](https://github.com/zongyi-li/fourier_neural_operator)).
  + `wavelet_convolution.py` contains functions for 1D, 2D, and 3D convolution in wavelet domain.
```

## Essential Python Libraries
Following packages are required to be installed to run the above codes:
  + [PyTorch](https://pytorch.org/)
  + [PyWavelets - Wavelet Transforms in Python](https://pywavelets.readthedocs.io/en/latest/)
  + [Wavelet Transforms in Pytorch](https://github.com/fbcotter/pytorch_wavelets)
  + [Wavelet Transform Toolbox](https://github.com/v0lta/PyTorch-Wavelet-Toolbox)
  + [Xarray-Grib reader (To read ERA5 data in section 5)](https://docs.xarray.dev/en/stable/getting-started-guide/installing.html?highlight=install)

Copy all the data in the folder 'data' and place the folder 'data' inside the same mother folder where the codes are present.	Incase, the location of the data are changed, the correct path should be given.

## Testing
For performing predictions on new inputs, one can use the 'WNO_testing_(.).py' codes given in the `Testing` folder in previous version. The trained models, that were used to produce results for the WNO paper can be found in the following link:
  > [Models](https://drive.google.com/drive/folders/1scfrpChQ1wqFu8VAyieoSrdgHYCbrT6T?usp=sharing)

However, these trained models are not compatible with the version 2.0.0 codes. Hence, new models need to be trained accordingly.

## Dataset
  + The training and testing datasets for the (i) Burgers equation with discontinuity in the solution field (section 4.1), (ii) 2-D Allen-Cahn equation (section 4.5), and (iii) Weakly-monthly mean 2m air temperature (section 5) are available in the following link:
    > [Dataset-1](https://drive.google.com/drive/folders/1scfrpChQ1wqFu8VAyieoSrdgHYCbrT6T?usp=sharing) \
The dataset for the Weakly and monthly mean 2m air temperature are downloaded from 'European Centre for Medium-Range Weather Forecasts (ECMEF)' database. For more information on the dataset one can browse the link 
    [ECMEF](https://www.ecmwf.int/en/forecasts/datasets/browse-reanalysis-datasets).
  + The datasets for (i) 1-D Burgers equation ('burgers_data_R10.zip'), (ii) 2-D Darcy flow equation in a rectangular domain ('Darcy_421.zip'), (iii) 2-D time-dependent Navier-Stokes equation ('ns_V1e-3_N5000_T50.zip'), are taken from the following link:
    > [Dataset-2](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)
  + The datasets for 2-D Darcy flow equation with a notch in triangular domain ('Darcy_Triangular_FNO.mat') and 1-D time-dependent wave advection equation are taken from the following link:
    > [Dataset-3](https://github.com/lu-group/deeponet-fno/tree/main/data)

## BibTex
If you use any part our codes, please cite us at,
```
@article{tripura2023wavelet,
  title={Wavelet Neural Operator for solving parametric partial differential equations in computational mechanics problems},
  author={Tripura, Tapas and Chakraborty, Souvik},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={404},
  pages={115783},
  year={2023},
  publisher={Elsevier}
}
```

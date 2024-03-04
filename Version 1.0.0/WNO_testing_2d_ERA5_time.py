"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
   
-- This code is for performing predictions on pre-trained models for
   weekly forecast of 2m air temperature (time-dependent problem).
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

import xarray as xr
from timeit import default_timer
from pytorch_wavelets import DWT, IDWT # (or import DWT, IDWT)

torch.manual_seed(0)
np.random.seed(0)

# %%

class WaveConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(WaveConv2d, self).__init__()

        """
        2D Wavelet layer. It does DWT, linear transform, and Inverse dWT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute single tree Discrete Wavelet coefficients using some wavelet
        dwt = DWT(J=2, mode='symmetric', wave='db4').to(device)
        x_ft, x_coeff = dwt(x)

        # Multiply relevant Wavelet modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x_ft.shape[-2], x_ft.shape[-1], device=x.device)
        out_ft = self.compl_mul2d(x_ft, self.weights1)
        # Multiply the finer wavelet coefficients
        x_coeff[-1][:,:,0,:,:] = self.compl_mul2d(x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
        
        # Return to physical space        
        idwt = IDWT(mode='symmetric', wave='db4').to(device)
        x = idwt((out_ft, x_coeff))
        return x

# %%
class WNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(WNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Wavelet layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain when required
        self.fc0 = nn.Linear(9, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = WaveConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = WaveConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = WaveConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = WaveConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 360, size_x), dtype=torch.float)    # latitudes
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(90, -90, size_y), dtype=torch.float)   # longitudes
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

# %%

PATH = 'data/ERA5_day_5years.grib'

ntrain = 270
ntest = 6

batch_size = 3
learning_rate = 0.001

epochs = 500
step_size = 50
gamma = 0.75

modes1 = 17
modes2 = 28
width = 20

r = 2**4
h = int(((721 - 1)/r))
s = int(((1441 - 1)/r))

T = 7
step = 1

# %%

ds = xr.open_dataset(PATH, engine='cfgrib')
data = np.array(ds["t2m"])
data = torch.tensor(data)
# data = data[:,:720,:]

Tn = 7*int(1937/7)
x_data = data[:-1, :, :]
y_data = data[1:, :, :]

x_data = x_data[:Tn, :, :]
y_data = y_data[:Tn, :, :]

x_data = x_data.reshape(1932,721,1440,1)
x_data = list(torch.split(x_data, int(1932/7), dim=0))
x_data = torch.cat((x_data), dim=3)

y_data = y_data.reshape(1932,721,1440,1)
y_data = list(torch.split(y_data, int(1932/7), dim=0))
y_data = torch.cat((y_data), dim=3)

# %%
x_train = x_data[:ntrain, ::r, ::r]
y_train = y_data[:ntrain, ::r, ::r]

x_test = y_data[-ntest:, ::r, ::r]
y_test = y_data[-ntest:, ::r, ::r]

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# %%

model = torch.load('model/model_wno_2d_ERA5_time')
print(count_params(model))

myloss = LpLoss(size_average=False)

# %%
pred0 = torch.zeros(y_test.shape)
index = 0
test_e = torch.zeros(y_test.shape[0])        
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
error = torch.zeros(y_test.shape[0],T)
with torch.no_grad():
    for xx, yy in test_loader:
        test_l2_step = 0
        test_l2_full = 0
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(1, y.size()[-3], y.size()[-2]), y.reshape(1, y.size()[-3], y.size()[-2]))
            error[index, t] = myloss(im.reshape(1, y.size()[-3], y.size()[-2]), y.reshape(1, y.size()[-3], y.size()[-2]))
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)
            
        pred0[index] = pred
        test_l2_step += loss.item()
        test_l2_full += myloss(pred.reshape(1, -1), yy.reshape(1, -1)).item()
        test_e[index] = test_l2_step
        
        print(index, test_l2_step/ ntest/ (T/step), test_l2_full/ ntest)
        index = index + 1
        
print('Mean Testing Error:', 100*torch.mean(test_e).numpy() / (T/step), '%')

# %%
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'

figure1 = plt.figure(figsize = (18, 16))
plt.subplots_adjust(hspace=0.05, wspace=0.18)
batch_no = 5
index = 0
for tvalue in range(10):
    if tvalue < 6: #(printing till Mon.-Sat.)
        ###
        plt.subplot(4,3, index+1)
        plt.imshow(y_test.numpy()[batch_no,:,:,tvalue], cmap='gist_ncar', interpolation='Gaussian')
        plt.title('Day-{}'.format(tvalue+1)); plt.xlabel('Longitude ($^{\circ}$)', fontweight='bold'); 
        plt.grid(True)
        if index == 0 or index == 3:
            plt.ylabel('Truth \n Latitude ($^{\circ}$)', fontweight='bold')
        else:
            plt.ylabel('Latitude ($^{\circ}$)', fontweight='bold')
        
        ###
        plt.subplot(4,3, index+1+6)
        plt.imshow(pred0[batch_no,:,:,tvalue], cmap='gist_ncar', interpolation='Gaussian')
        plt.title('Day-{} (error={:0.4f}%)'.format(tvalue+1,100*error[batch_no,tvalue])); 
        plt.xlabel('Longitude ($^{\circ}$)', fontweight='bold');
        plt.grid(True)
        if index == 0 or index == 3:
            plt.ylabel('Prediction \n Latitude ($^{\circ}$)', fontweight='bold')
        else:
            plt.ylabel('Latitude ($^{\circ}$)', fontweight='bold')
        index = index + 1

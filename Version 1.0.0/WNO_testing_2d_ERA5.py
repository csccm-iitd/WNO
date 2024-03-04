"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
   
-- This code is for performing predictions on pre-trained models for
   forecast of monthly averaged 2m air temperature (time-independent problem).
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import xarray as xr
from timeit import default_timer
from utilities3 import *
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
        dwt = DWT(J=5, mode='symmetric', wave='db4').to(device)
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
        self.padding = 1 # pad the domain when required
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

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

PATH = 'data/ERA5_temp.grib'

ntrain = 460
ntest = 50

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 25
gamma = 0.75

modes1 = 10
modes2 = 14
width = 64

r = 6
h = int((721 - 1)/r+1)
s = int((1441 - 1)/r+1)

# %%

ds = xr.open_dataset(PATH, engine='cfgrib')
data = np.array(ds["t2m"])
data = torch.tensor(data)
# data = data[:, :720, :]
data = F.pad(data, [0,1]) # pad last dimension to make it periodic

x_train = data[:ntrain, ::r, ::r]
y_train = data[:ntrain, ::r, ::r]

x_test = data[-ntest:, ::r, ::r]
y_test = data[-ntest:, ::r, ::r]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain,h,s,1)
x_test = x_test.reshape(ntest,h,s,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# %%

model = torch.load('model/model_wno_2d_ERA5_t2m')
print(count_params(model))
myloss = LpLoss(size_average=False)
y_normalizer.cuda()

# %%
pred = torch.zeros(y_test.shape)
index = 0
test_e = torch.zeros(y_test.shape[0])
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.to(device), y.to(device)

        out = model(x).reshape(h, s)
        out = y_normalizer.decode(out)
        pred[index] = out

        test_l2 += myloss(out.reshape(1, h, s), y.reshape(1, h, s)).item()
        test_e[index] = test_l2
        print(index, test_l2)
        index = index + 1
        
print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')
   
# %%
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 12

figure1 = plt.figure(figsize = (12, 13))
plt.subplots_adjust(hspace=0.01, wspace=0.25)
index = 1
for value in range(y_test.shape[0]):
    if value % 22 == 1 and value != 1:
        ###
        img = y_test[value,:,:-1].cpu().numpy()
        plt.subplot(3,2, index)
        plt.imshow(img, cmap='nipy_spectral', extent=[0,360,-90,+90])
        plt.xlabel('Longitude ($^{\circ}$)'); plt.ylabel('Lattitude ($^{\circ}$)')
        plt.grid(True)
        if index==1:
            plt.title('Truth: Feb 2019, 1st'); 
        else:
            plt.title('Truth: Feb 2021, 1st')
        
        ###
        plt.subplot(3,2, index+2)
        plt.imshow(pred[value,:,:-1], cmap='nipy_spectral', extent=[0,360,-90,+90])
        plt.xlabel('Longitude ($^{\circ}$)'); plt.ylabel('Lattitude ($^{\circ}$)')
        plt.grid(True)
        if index==1:
            plt.title('Identification - Feb 2019, 1st \n (Error-{:0.4f})'.format(100*test_e[value].numpy()))
        else:
            plt.title('Identification - Feb 2021, 1st \n (Error-{:0.4f})'.format(100*test_e[value].numpy()))
        
        index = index + 1
   
"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
   
-- This code is for performing predictions on pre-trained models for
   1-D Advection equation (time-dependent problem).
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import pywt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from timeit import default_timer
from utilities3 import *
from pytorch_wavelets import DWT1D, IDWT1D

torch.manual_seed(0)
np.random.seed(0)

# %%

class WaveConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(WaveConv1d, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute single tree Discrete Wavelet coefficients using some wavelet 
        dwt = DWT1D(wave='db6', J=3, mode='symmetric').to(device)
        x_ft, x_coeff = dwt(x)
        
        # Multiply the final low pass and high pass coefficients
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1],  device=x.device)
        out_ft = self.compl_mul1d(x_ft, self.weights1)
        x_coeff[-1] = self.compl_mul1d(x_coeff[-1], self.weights1)
        
        idwt = IDWT1D(wave='db6', mode='symmetric').to(device)
        x = idwt((out_ft, x_coeff))
        return x

class WNO1d(nn.Module):
    def __init__(self, modes, width):
        super(WNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Wavelet layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 2 # pad the domain when required
        self.fc0 = nn.Linear(40, self.width) # input channel is 2: (a(x), x)

        self.conv0 = WaveConv1d(self.width, self.width, self.modes1)
        self.conv1 = WaveConv1d(self.width, self.width, self.modes1)
        self.conv2 = WaveConv1d(self.width, self.width, self.modes1)
        self.conv3 = WaveConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) 

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

        # x = x[..., :-self.padding] 
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 100, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


# %%

ntrain = 1000
ntest = 100
s = 40

batch_size = 25 # 50
learning_rate = 0.001

epochs = 500
scheduler_step = 50
scheduler_gamma = 0.5

modes = 14
width = 64
T = 39
step = 1

# %%

data = np.load('data/train_IC2.npz')
x, t, u_train = data["x"], data["t"], data["u"]  # N x nt x nx
x_train = u_train[:ntrain, :-1, :]  # N x nx
y_train = u_train[:ntrain, 1:, :] # one step ahead,
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_train = x_train.permute(0,2,1)
y_train = y_train.permute(0,2,1)

data = np.load('data/test_IC2.npz')
x, t, u_test = data["x"], data["t"], data["u"]  # N x nt x nx
x_test = u_test[:ntest, :-1, :]  # N x nx
y_test = u_test[:ntest, 1:, :] # one step ahead,
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)
x_test = x_test.permute(0,2,1)
y_test = y_test.permute(0,2,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

# %%

# model
model = torch.load('model/model_wno_1d_advection_III_time')
print(count_params(model))

myloss = LpLoss(size_average=False)

# %%
pred0 = torch.zeros(y_test.shape)
index = 0      
test_e = torch.zeros(y_test.shape)     
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
with torch.no_grad():
    for xx, yy in test_loader:
        test_l2_step = 0
        test_l2_full = 0
        loss = 0
        mse = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(1, y.size()[-3], y.size()[-2]), y.reshape(1, y.size()[-3], y.size()[-2]))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)
            
        pred0[index] = pred
        test_l2_step += loss.item()
        test_l2_full += myloss(pred.reshape(1, -1), yy.reshape(1, -1)).item()
        mse += F.mse_loss(pred.reshape(1, -1), yy.reshape(1, -1), reduction='mean')
        test_e[index] = test_l2_step
        
        print(index, test_l2_step, test_l2_full, mse.cpu().numpy())
        index = index + 1

print('Mean Testing Error:', 100*torch.mean(test_e).numpy() /(T/step), '%')

# %%
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 16

figure1 = plt.figure(figsize = (18, 14))
figure1.text(0.03,0.17,'\n Error', rotation=90, color='purple', fontsize=20)
figure1.text(0.03,0.34,'\n Prediction', rotation=90, color='green', fontsize=20)
figure1.text(0.03,0.57,'\n Truth', rotation=90, color='red', fontsize=20)
figure1.text(0.03,0.75,'Initial \n Condition', rotation=90, color='b', fontsize=20)
plt.subplots_adjust(wspace=0.7)
index = 0
for value in range(y_test.shape[0]):
    if value % 23 == 1 and value != 1:
        print(value)
        plt.subplot(4,4, index+1)
        plt.plot(np.linspace(0,1,39),x_test[value,0,:], linewidth=2, color='blue')
        plt.title('IC-{}'.format(index+1), color='b', fontsize=20, fontweight='bold')
        plt.xlabel('x', fontweight='bold'); plt.ylabel('u(x,0)', fontweight='bold'); 
        plt.margins(0)
        ax = plt.gca();
        ratio = 0.9
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)
        
        plt.subplot(4,4, index+1+4)
        plt.imshow(y_test[value,:,:], cmap='Spectral', extent=[0,1,0,1], interpolation='Gaussian')
        plt.xlabel('x', fontweight='bold'); plt.ylabel('t', fontweight='bold', color='m', fontsize=20); 
        plt.colorbar(fraction=0.045)
        
        plt.subplot(4,4, index+1+8)
        plt.imshow(pred0[value,:,:], cmap='Spectral', extent=[0,1,0,1], interpolation='Gaussian')
        plt.xlabel('x', fontweight='bold'); plt.ylabel('t', fontweight='bold', color='m', fontsize=20); 
        plt.colorbar(fraction=0.045)
        
        plt.subplot(4,4, index+1+12)
        plt.imshow(np.abs(y_test[value,:,:]-pred0[value,:,:]), cmap='jet', extent=[0,1,0,1],
                   vmax= 1, interpolation='Gaussian')
        plt.xlabel('x', fontweight='bold'); plt.ylabel('t', fontweight='bold', color='m', fontsize=20); 
        plt.colorbar(fraction=0.045)
        
        plt.margins(0)
        index = index + 1
 
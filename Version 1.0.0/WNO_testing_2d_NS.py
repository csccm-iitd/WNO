"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
   
-- This code is for performing predictions on pre-trained models for
   2-D Darcy Navier-Stokes equation (time-dependent problem).
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utilities3 import *

from timeit import default_timer
from pytorch_wavelets import DWT, IDWT # (or import DWT, IDWT)

torch.manual_seed(0)
np.random.seed(0)

# %%

class WNOConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(WNOConv2d_fast, self).__init__()

        """
        2D Wavelet layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Wavelet modes to multiply, at most floor(N/2) + 1
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
        # dwt = DWT(J=3, mode='symmetric', wave='db6').to(device)
        dwt = DWT(J=3, mode='symmetric', wave='db4').to(device)

        x_ft, x_coeff = dwt(x)
        # print(x_ft.shape)

        # Multiply relevant Wavelet modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x_ft.shape[-2], x_ft.shape[-1], device=x.device)
        out_ft = self.compl_mul2d(x_ft, self.weights1)
        # Multiply the finer wavelet coefficients
        x_coeff[-1][:,:,0,:,:] = self.compl_mul2d(x_coeff[-1][:,:,0,:,:].clone(), self.weights2)
        
        # Return to physical space        
        idwt = IDWT(mode='symmetric', wave='db4').to(device)
        x = idwt((out_ft, x_coeff))
        
        return x

class WNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(WNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Wavelet layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain when required
        self.fc0 = nn.Linear(12, self.width)
        # input channel is 12: the solution of the previous 10 timesteps +
        # 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = WNOConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = WNOConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = WNOConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = WNOConv2d_fast(self.width, self.width, self.modes1, self.modes2)
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
        # x = F.pad(x, [0,self.padding, 0,self.padding])

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

        # x = x[..., :-self.padding, :-self.padding] 
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


# %%

TRAIN_PATH = 'data/ns_V1e-3_N5000_T50.mat'

ntrain = 1000
ntest = 20

modes = 14
width = 26 

batch_size = 20 
batch_size2 = batch_size

epochs = 800
learning_rate = 0.001
scheduler_step = 50
scheduler_gamma = 0.75

sub = 1
S = 64
T_in = 10
T = 10
step = 1

# %%

reader = MatReader(TRAIN_PATH)
data = reader.read_field('u')
train_a = data[:ntrain,::sub,::sub,:T_in]
train_u = data[:ntrain,::sub,::sub,T_in:T+T_in]

test_a = data[-ntest:,::sub,::sub,:T_in]
test_u = data[-ntest:,::sub,::sub,T_in:T+T_in]

print(train_u.shape)
print(test_u.shape)
assert (S == train_u.shape[-2])
assert (T == train_u.shape[-1])

train_a = train_a.reshape(ntrain,S,S,T_in)
test_a = test_a.reshape(ntest,S,S,T_in)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

# %%

model = torch.load('model/model_wno_2d_navier_stokes')
print(count_params(model))

myloss = LpLoss(size_average=False)

# %%
pred0 = torch.zeros(test_u.shape)
index = 0
test_e = torch.zeros(test_u.shape)        
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)

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
        
print('Mean Testing Error:', 100*torch.mean(test_e).numpy() / ntest/ (T/step), '%')

# %% 
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 14

figure1 = plt.figure(figsize = (18, 14))
figure1.text(0.04,0.17,'\n Error', rotation=90, color='purple', fontsize=20)
figure1.text(0.04,0.34,'\n Prediction', rotation=90, color='green', fontsize=20)
figure1.text(0.04,0.57,'\n Truth', rotation=90, color='red', fontsize=20)
figure1.text(0.04,0.75,'Initial \n Condition', rotation=90, color='b', fontsize=20)
plt.subplots_adjust(wspace=0.7)
index = 0
for value in range(test_u.shape[-1]):
    if value % 3 == 0:
        print(value)
        plt.subplot(4,4, index+1)
        plt.imshow(test_a.numpy()[15,:,:,0], cmap='jet', extent=[0,1,0,1], interpolation='Gaussian')
        plt.title('t={}s'.format(value+10), color='b', fontsize=18, fontweight='bold')
        plt.xlabel('x',fontweight='bold'); plt.ylabel('y',fontweight='bold')
        plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
        
        plt.subplot(4,4, index+1+4)
        plt.imshow(test_u[15,:,:,value], cmap='jet', extent=[0,1,0,1], interpolation='Gaussian')
        plt.colorbar(fraction=0.045)
        plt.xlabel('x',fontweight='bold'); plt.ylabel('y',fontweight='bold')
        plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
        
        plt.subplot(4,4, index+1+8)
        plt.imshow(pred0[15,:,:,value], cmap='jet', extent=[0,1,0,1], interpolation='Gaussian')
        plt.colorbar(fraction=0.045)
        plt.xlabel('x',fontweight='bold'); plt.ylabel('y',fontweight='bold')
        plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
        
        plt.subplot(4,4, index+1+12)
        plt.imshow(np.abs(test_u[15,:,:,value]-pred0[15,:,:,value]), cmap='jet', extent=[0,1,0,1], interpolation='Gaussian')
        plt.xlabel('x', fontweight='bold'); plt.ylabel('y', fontweight='bold'); 
        plt.colorbar(fraction=0.045,format='%.0e')
        
        plt.margins(0)
        index = index + 1

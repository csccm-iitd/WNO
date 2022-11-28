"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
   
-- This code is for performing predictions on pre-trained models for
   1-D Burger's equation (time-independent problem).
"""

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
    def __init__(self, in_channels, out_channels, level1):
        super(WaveConv1d, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform, linear transform, and
        Inverse Wavelet Transform.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level1 = level1  

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.level1+6))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.level1+6))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        #Compute single tree Discrete Wavelet coefficients using some wavelet     
        dwt = DWT1D(wave='db6', J=self.level1, mode='symmetric').to(device)
        x_ft, x_coeff = dwt(x)
        
        # Multiply the final low pass and high pass coefficients
        out_ft = self.compl_mul1d(x_ft, self.weights1)
        x_coeff[-1] = self.compl_mul1d(x_coeff[-1], self.weights2)
        
        idwt = IDWT1D(wave='db6', mode='symmetric').to(device)
        x = idwt((out_ft, x_coeff))        
        return x

class WNO1d(nn.Module):
    def __init__(self, level, width):
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

        self.level1 = level
        self.width = width
        self.padding = 2 # pad the domain when required
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)

        self.conv0 = WaveConv1d(self.width, self.width, self.level1)
        self.conv1 = WaveConv1d(self.width, self.width, self.level1)
        self.conv2 = WaveConv1d(self.width, self.width, self.level1)
        self.conv3 = WaveConv1d(self.width, self.width, self.level1)
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
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


# %%

ntrain = 1000
ntest = 100

sub = 2**3 #subsampling rate
h = 2**13 // sub #total grid size divided by the subsampling rate
s = h

batch_size = 10
learning_rate = 0.001

epochs = 800
step_size = 50
gamma = 0.75

level = 8
width = 64

# %%

# Data is of the shape (number of samples, grid size)
dataloader = MatReader('data/burgers_data_R10.mat')
x_data = dataloader.read_field('a')[:,::sub]
y_data = dataloader.read_field('u')[:,::sub]

x_train = x_data[:ntrain,:]
y_train = y_data[:ntrain,:]
x_test = x_data[-ntest:,:]
y_test = y_data[-ntest:,:]

x_train = x_train.reshape(ntrain,s,1)
x_test = x_test.reshape(ntest,s,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                          batch_size=batch_size, shuffle=False)

# %%

model = torch.load('model/model_wno_1d_burgers')
print(count_params(model))

myloss = LpLoss(size_average=False)

# %%
pred = torch.zeros(y_test.shape)
index = 0
test_e = torch.zeros(y_test.shape)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
with torch.no_grad():
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.to(device), y.to(device)

        out = model(x).view(-1)
        pred[index] = out

        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        test_e[index] = test_l2
        print(index, test_l2)
        index = index + 1

print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')

# %%

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 16

figure1 = plt.figure(figsize = (12, 8))
plt.subplots_adjust(hspace=0.4)
for i in range(y_test.shape[0]):
    if i % 23 == 1:
        plt.subplot(2,1,1)
        plt.plot(np.linspace(0,1,1024),x_test[i, :].numpy())
        plt.title('(a) I.C.')
        plt.xlabel('x', fontsize=20, fontweight='bold')
        plt.ylabel('u(x,0)', fontsize=20, fontweight='bold')
        plt.grid(True)
        plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
        plt.margins(0)
        
        plt.subplot(2,1,2)
        plt.plot(np.linspace(0,1,1024),y_test[i, :].numpy())
        plt.plot(np.linspace(0,1,1024),pred[i,:], ':k') 
        plt.title('(b) Solution')
        plt.legend(['Truth', 'Prediction'], ncol=2, loc=3, fontsize=20)
        plt.xlabel('x', fontsize=20, fontweight='bold')
        plt.ylabel('u(x,1)', fontsize=20, fontweight='bold')
        plt.grid(True)
        plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
        plt.margins(0)

"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
   
-- This code is for 1-D Burger's equation (time-independent problem).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from timeit import default_timer
from utils import *
from wavelet_convolution import WaveConv1d

torch.manual_seed(0)
np.random.seed(0)

# %%
""" The forward operation """
class WNO1d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, padding=0):
        super(WNO1d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x) = g(K.v + W.v)(x).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 2-channel tensor, Initial condition and location (a(x), x)
              : shape: (batchsize * x=s * c=2)
        Output: Solution of a later timestep (u(x))
              : shape: (batchsize * x=s * c=1)
              
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : scalar, signal length
        wavelet: string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: scalar (for 1D), right support of 1D domain
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet = wavelet
        self.in_channel = in_channel
        self.grid_range = grid_range 
        self.padding = padding
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 2: (a(x), x)
        for i in range( self.layers ):
            self.conv.append( WaveConv1d(self.width, self.width, self.level, self.size, self.wavelet) )
            self.w.append( nn.Conv1d(self.width, self.width, 1) )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)              # Shape: Batch * x * Channel
        x = x.permute(0, 2, 1)       # Shape: Batch * Channel * x
        if self.padding != 0:
            x = F.pad(x, [0,self.padding]) 
        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:   # Final layer has no activation    
                x = F.mish(x)        # Shape: Batch * Channel * x 
                
        if self.padding != 0:
            x = x[..., :-self.padding] 
        x = x.permute(0, 2, 1)       # Shape: Batch * x * Channel
        x = F.gelu( self.fc1(x) )    # Shape: Batch * x * Channel
        x = self.fc2(x)              # Shape: Batch * x * Channel
        return x

    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, self.grid_range, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


# %%
""" Model configurations """

PATH = 'data/burgers_data_R10.mat'
ntrain = 1000
ntest = 100

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 50   # weight-decay step size
gamma = 0.5      # weight-decay rate

wavelet = 'db6'  # wavelet basis function
level = 8        # lavel of wavelet decomposition
width = 64       # uplifting dimension
layers = 4       # no of wavelet layers

sub = 2**3       # subsampling rate
test_sub = 2**2
h = 2**13 // sub # total grid size divided by the subsampling rate
grid_range = 1
in_channel = 2   # (a(x), x) for this case

# %%
""" Read data """

dataloader = MatReader(PATH)
x_data_1024 = dataloader.read_field('a')[:,::sub]
y_data_1024 = dataloader.read_field('u')[:,::sub]

x_data_2048 = dataloader.read_field('a')[:,::test_sub]
y_data_2048 = dataloader.read_field('u')[:,::test_sub]

x_train_1024, y_train_1024 = x_data_1024[:ntrain,:], y_data_1024[:ntrain,:]
x_test_1024, y_test_1024 = x_data_1024[-ntest:,:], y_data_1024[-ntest:,:]

x_train_2048, y_train_2048 = x_data_2048[:ntrain,:], y_data_2048[:ntrain,:]
x_test_2048, y_test_2048 = x_data_2048[-ntest:,:], y_data_2048[-ntest:,:]

x_train_1024 = x_train_1024[:, :, None]
x_test_1024 = x_test_1024[:, :, None]

x_train_2048 = x_train_2048[:, :, None]
x_test_2048 = x_test_2048[:, :, None]

train_loader_1024 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_1024, y_train_1024),
                                           batch_size=batch_size, shuffle=True)
test_loader_1024 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_1024, y_test_1024),
                                          batch_size=batch_size, shuffle=False)

train_loader_2048 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_2048, y_train_2048),
                                           batch_size=batch_size, shuffle=True)
test_loader_2048 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_2048, y_test_2048),
                                          batch_size=batch_size, shuffle=False)

# %%
""" The model definition """
model = WNO1d(width=width, level=level, layers=layers, size=h, wavelet=wavelet,
              in_channel=in_channel, grid_range=grid_range).to(device)
print(count_params(model))

model.load_state_dict(torch.load('model/WNO_burgers', map_location=device).state_dict())
myloss = LpLoss(size_average=False)

# %%
""" Prediction """
pred_1024, pred_2048 = [], []
test_e_1024, test_e_2048 = [], []
with torch.no_grad():
    
    index = 0
    for x, y in test_loader_1024:
        test_l2 = 0 
        x, y = x.to(device), y.to(device)

        out = model(x)
        test_l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        test_e_1024.append( test_l2/batch_size )
        pred_1024.append( out.cpu() )
        print("Batch-{}, Train and Test at {}, Test-loss-{:0.6f}".format( index, 2**13//sub, test_l2/batch_size ))
        index += 1
    
    index = 0
    for x, y in test_loader_2048:
        test_l2 = 0 
        x, y = x.to(device), y.to(device)

        out = model(x)
        test_l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        test_e_2048.append( test_l2/batch_size )
        pred_2048.append( out.cpu() )
        print("Batch-{}, Train at {} and Test at {}, Test-loss-{:0.6f}".format( index, 2**13//sub,
                                                                               2**13//test_sub, test_l2/batch_size ))
        index += 1

pred_1024 = torch.cat(( pred_1024 ))
pred_2048 = torch.cat(( pred_2048 ))
test_e_1024 = torch.tensor(( test_e_1024 ))  
test_e_2048 = torch.tensor(( test_e_2048 ))  

print('\nMean Error: Resolution-1024-{:0.4f}, Resolution-2048-{:0.4f}'
      .format(100*torch.mean(test_e_1024).numpy(), 100*torch.mean(test_e_2048).numpy()))

# %%
plt.rcParams['font.family'] = 'Times New Roman' 
plt.rcParams['font.size'] = 14
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

colormap = plt.cm.rainbow  
colors = [colormap(i) for i in np.linspace(0, 1, 5)]

""" Plotting """ 
figure7, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), dpi=300)
plt.subplots_adjust(hspace=0.35)
index = 0
for i in range(ntest):
    if i % 20 == 0:
        ax[0].plot(y_test_1024[i, :].cpu().numpy(), color=colors[index], label='Truth-s{}'.format(i))
        ax[0].plot(pred_1024[i,:].cpu().numpy(), '--', color=colors[index], label='WNO-s{}'.format(i))
        ax[0].grid(True, alpha=0.35)
        ax[0].legend(ncol=5, columnspacing=0.5, labelspacing=0.25, handletextpad=0.25, borderpad=0.15)
        ax[0].margins(0)
        ax[0].set_title('Train at resolution 1024', fontweight='bold', fontsize=plt.rcParams['font.size']*1.2)
        
        ax[1].plot(y_test_2048[i, :].cpu().numpy(), color=colors[index], label='Truth-s{}'.format(i))
        ax[1].plot(pred_2048[i,:].cpu().numpy(), '--', color=colors[index], label='WNO-s{}'.format(i))
        ax[1].grid(True, alpha=0.35)
        ax[1].legend(ncol=5, columnspacing=0.5, labelspacing=0.25, handletextpad=0.25, borderpad=0.15)
        ax[1].margins(0)
        ax[1].set_title('Test at resolution 2048', fontweight='bold', fontsize=plt.rcParams['font.size']*1.2)
        index += 1
ax[0].set_xlabel('Space')
ax[1].set_xlabel('Space')
ax[0].set_ylabel('$u(x,1)$')
ax[1].set_ylabel('$u(x,1)$')
figure7.suptitle('Superresolution in Burgers equation', fontweight='bold', y=0.95, fontsize=plt.rcParams['font.size']*1.4)
plt.show()

# figure7.savefig('Burgers_prediction.png', format='png', dpi=300, bbox_inches='tight')

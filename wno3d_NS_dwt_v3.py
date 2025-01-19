"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet neural operator: a neural 
   operator for parametric partial differential equations. arXiv preprint arXiv:2205.02191.
   
This code is for 2-D Navier-Stokes equation (2D time-dependent problem) using Time as third axis.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from timeit import default_timer
from utils import *
from wavelet_convolution_v3 import WaveConv3d

torch.manual_seed(0)
np.random.seed(0)
device = ('cuda' if torch.cuda.is_available() else 'cpu')

# %%
""" The forward operation """
class WNO3d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_chanel, grid_range, omega, padding=0):
        super(WNO3d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 4-channel tensor, Input at t0 and location (a(x,y,t), t, x, y)
              : shape: (batchsize * t=time * x=width * x=height * c=4)
        Output: Solution of a later timestep (u(x, T_in+1))
              : shape: (batchsize * t=time * x=width * x=height * c=1)
        
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : list with 3 elements (for 3D), the 3D volume size
        wavelet   : string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: list with 3 elements (for 3D), right supports of the 3D domain
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.size = size
        self.wavelet = wavelet
        self.omega = omega
        self.layers = layers
        self.grid_range = grid_range 
        self.padding = padding
                
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        self.fc0 = nn.Linear(in_chanel, self.width) # input channel is 3: (a(x, y), x, y)
        for i in range(self.layers):
            self.conv.append(WaveConv3d(self.width, self.width, self.level, size=self.size, 
                                        wavelet=self.wavelet, omega=self.omega))
            self.w.append(nn.Conv3d(self.width, self.width, 1))

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)                 # Shape: Batch * x * y * z * Channel
        x = x.permute(0, 4, 3, 1, 2)    # Shape: Batch * Channel * z * x * y 
        if self.padding != 0:
            x = F.pad(x, [0,self.padding, 0,self.padding, 0,self.padding]) # do padding, if required
        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.mish(x)                # Shape: Batch * Channel * x * y
            
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding, :-self.padding] # remove padding, when required
        x = x.permute(0, 3, 4, 2, 1)        # Shape: Batch * x * y * z * Channel 
        x = self.fc2(F.mish(self.fc1(x)))   # Shape: Batch * x * y * z 
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, self.grid_range[2], size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


# %%
""" Model configurations """

PATH = '/home/user/Desktop/Papers_codes/P3_WNO/WNO-master/data/ns_V1e-3_N5000_T50.mat'
# PATH = 'data/ns_V1e-4_N10000_T30.mat'

ntrain = 1000
ntest = 100

batch_size = 10
learning_rate = 0.001

epochs = 500
step_size = 50   # weight-decay step size
gamma = 0.5      # weight-decay rate

wavelet = 'db2'  # wavelet basis function
level = 1        # lavel of wavelet decomposition
width = 32       # uplifting dimension
layers = 4       # no of wavelet layers

sub = 1          # subsampling rate
h = 64           # total grid size divided by the subsampling rate
grid_range = [1, 1, 1]
in_channel = 13  # input channel is 12: (10 for a(x,t1-t10), 2 for x)

T_in = 10
T = 20           # No of prediction steps
step = 1         # Look-ahead step size

# %%
""" Read data """
reader = MatReader(PATH)
data = reader.read_field('u')
train_a = data[:ntrain,::sub,::sub,:T_in]
train_u = data[:ntrain,::sub,::sub,T_in:T+T_in]

test_a = data[-ntest:,::sub,::sub,:T_in]
test_u = data[-ntest:,::sub,::sub,T_in:T+T_in]

a_normalizer = UnitGaussianNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

y_normalizer = UnitGaussianNormalizer(train_u)
train_u = y_normalizer.encode(train_u)

train_a = train_a.reshape(ntrain,h,h,1,T_in).repeat([1,1,1,T,1])
test_a = test_a.reshape(ntest,h,h,1,T_in).repeat([1,1,1,T,1])

# %%
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u),
                                          batch_size=batch_size, shuffle=False)

# %%
""" The model definition """
model = WNO3d(width, level, layers=layers, size=[T, h, h], wavelet=wavelet,
              in_chanel=in_channel, grid_range=grid_range, omega=6).to(device)
print(count_params(model))

""" Training and testing """
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

epoch_loss = torch.zeros(epochs)
myloss = LpLoss(size_average=False)
y_normalizer.to(device)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x).squeeze(-1)
        
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
        
        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss = myloss(out.view(out.shape[0],-1), y.view(y.shape[0],-1))
        
        loss.backward()
        optimizer.step()
        train_mse += mse.item()
        train_l2 += loss.item()
    
    scheduler.step()
    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            out = model(x).squeeze(-1)
            out = y_normalizer.decode(out)

            test_l2 += myloss(out.view(out.shape[0],-1), y.view(y.shape[0],-1)).item()
        
    train_mse /= len(train_loader)
    train_l2/= ntrain
    epoch_loss[ep] = train_l2
    test_l2 /= ntest
    t2 = default_timer()
    print("Epoch-{}, Time-{:0.4f}, Train-MSE-{:0.4f}, Train-L2-{:0.4f}, Test-L2-{:0.4f}"
          .format(ep, t2-t1, train_mse, train_l2, test_l2))

# %%
""" Prediction """
prediction = []
test_e = []     
with torch.no_grad():
    index = 0
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.to(device), y.to(device)

        out = model(x).squeeze(-1)
        out = y_normalizer.decode(out)

        test_l2 = myloss(out.view(out.shape[0],-1), y.view(y.shape[0],-1)).item()
         
        test_e.append( test_l2/batch_size )
        prediction.append( out.cpu() )
        print("Batch-{}, Test-loss-{:0.6f}".format( index, test_l2/batch_size ))
        index += 1
    
prediction = torch.cat(( prediction ))
test_e = torch.tensor((test_e))  
print('Mean Error:', 100*torch.mean(test_e).numpy())
  
# %%
""" Plotting """  
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 14

snapshots = np.arange(0,T,4)

figure1, ax = plt.subplots(nrows=3, ncols=len(snapshots)+1, figsize = (20, 8), dpi=100)
plt.subplots_adjust(hspace=0.4)
sample = 15

ax[0, 0].imshow(test_a.numpy()[sample,:,:,0,0], cmap='jet', extent=[0,1,0,1], interpolation='Gaussian')
ax[0, 0].set_title('t={}s'.format(0), color='b', fontsize=18, fontweight='bold')
ax[0, 0].set_ylabel('IC', rotation=90, color='r', fontsize=20)

for ii in range(1,3):
    ax[ii,0].set_axis_off()
    
for value in range(len(snapshots)):
    cmin = torch.min(test_u[sample,:,:,snapshots[value]])
    cmax = torch.max(test_u[sample,:,:,snapshots[value]])
    
    ax[0, value+1].imshow(test_u[sample,:,:,snapshots[value]], cmap='jet', extent=[0,1,0,1],
                          vmin=cmin, vmax=cmax, interpolation='Gaussian')
    
    ax[1, value+1].imshow(prediction[sample,:,:,snapshots[value]], cmap='jet', extent=[0,1,0,1],
                          vmin=cmin, vmax=cmax, interpolation='Gaussian')
    
    im = ax[2, value+1].imshow(np.abs(test_u[sample,:,:,snapshots[value]]-prediction[sample,:,:,snapshots[value]]),
                              vmin=0, vmax=0.1, cmap='jet', extent=[0,1,0,1], interpolation='Gaussian')
    plt.colorbar(im, ax=ax[2, value+1])
    ax[2, value+1].set_xticks([])
    ax[2, value+1].set_yticks([])
    
    if value == 1:
        ax[0, value].set_ylabel('Truth', rotation=90, color='g', fontsize=20)
        ax[1, value].set_ylabel('Prediction', rotation=90, color='b', fontsize=20)
        ax[2, value].set_ylabel('Error', rotation=90, color='purple', fontsize=20, labelpad=35)
plt.show()

# %%
"""
For saving the trained model and prediction data
"""
torch.save(model, 'model/WNO_navier_stokes_3D_v2p1')
scipy.io.savemat('results/wno_results_navier_stokes_3D_v2p1.mat', mdict={'test_a':test_a.cpu().numpy(),
                                                                         'test_u':test_u.cpu().numpy(),
                                                                         'prediction':prediction.cpu().numpy(),  
                                                                         'test_e':test_e.cpu().numpy()})

"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
   
-- This code is for weekly forecast of 2m air temperature (time-dependent problem).
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from utils import *

import xarray as xr
from timeit import default_timer
from wavelet_convolution import WaveConv2dCwt

torch.manual_seed(0)
np.random.seed(0)

# %%
""" The forward operation """
class WNO2d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, xgrid_range, ygrid_range, padding=0):
        super(WNO2d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 3-channel tensor, Initial input and location (a(x,y), x,y)
              : shape: (batchsize * x=width * x=height * c=3)
        Output: Solution of a later timestep (u(x,y))
              : shape: (batchsize * x=width * x=height * c=1)
              
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : list with 2 elements (for 2D), image size
        wavelet: list of strings for 2D, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: list with 2 elements (for 2D), right supports of 2D domain
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.layers = layers
        self.size = size
        self.wavelet1 = wavelet[0]
        self.wavelet2 = wavelet[1]
        self.in_channel = in_channel
        self.xgrid_range = xgrid_range
        self.ygrid_range = ygrid_range
        self.padding = padding
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        
        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 3: (a(x, y), x, y)
        for i in range( self.layers ):
            self.conv.append( WaveConv2dCwt(self.width, self.width, self.level, self.size,
                                            self.wavelet1, self.wavelet2) )
            self.w.append( nn.Conv2d(self.width, self.width, 1) )
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)    
        x = self.fc0(x)                      # Shape: Batch * x * y * Channel
        x = x.permute(0, 3, 1, 2)            # Shape: Batch * Channel * x * y
        if self.padding != 0:
            x = F.pad(x, [0,self.padding, 0,self.padding]) 
        
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x) 
            if index != self.layers - 1:     # Final layer has no activation    
                x = F.mish(x)                # Shape: Batch * Channel * x * y
                
        if self.padding != 0:
            x = x[..., :-self.padding, :-self.padding]     
        x = x.permute(0, 2, 3, 1)            # Shape: Batch * x * y * Channel
        x = F.gelu( self.fc1(x) )            # Shape: Batch * x * y * Channel
        x = self.fc2(x)                      # Shape: Batch * x * y * Channel
        return x
    
    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(self.xgrid_range[0], self.xgrid_range[1], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(self.ygrid_range[0], self.ygrid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device) 

# %%
""" Model configurations """

PATH = 'data/ERA5_daily_average_5years.grib'
ntrain = 270
ntest = 6

batch_size = 3
learning_rate = 0.001

epochs = 500
step_size = 50
gamma = 0.75

wavelet = ['near_sym_b', 'qshift_b']  # wavelet basis function
level = 2        # lavel of wavelet decomposition
width = 20       # uplifting dimension
layers = 4       # no of wavelet layers

sub = 2**4 # 2**4 for 4^o, # 2**3 for 2^o
h = int(((721 - 1)/sub))
s = int(((1441 - 1)/sub))

xgrid_range = [0, 360]          # The grid boundary in x direction
ygrid_range = [90, -90]          # The grid boundary in y direction
in_channel = 9  # input channel is 12: (10 for a(x,t1-t10), 2 for x)

T = 7
step = 1

# %%
""" Read data """

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
x_train = x_data[:ntrain, ::sub, ::sub, :]
y_train = y_data[:ntrain, ::sub, ::sub, :]

x_test = y_data[-ntest:, ::sub, ::sub, :]
y_test = y_data[-ntest:, ::sub, ::sub, :]

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                          batch_size=batch_size, shuffle=False)

# %%
""" The model definition """
model = WNO2d(width=width, level=level, layers=layers, size=[h,s], wavelet=wavelet,
              in_channel=in_channel, xgrid_range=xgrid_range, ygrid_range=ygrid_range, padding=2).to(device)
print(count_params(model))

""" Training and testing """
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_loss = torch.zeros(epochs)
test_loss = torch.zeros(epochs)
myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_batch = 0
    for xx, yy in train_loader:
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        
        for t in range(0, T, step):
            y = yy[..., t:t + step] # t:t+step, retains the third dimension,

            im = model(xx)            
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
            
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            xx = torch.cat((xx[..., step:], im), dim=-1)

        train_l2_step += loss.item()
        train_l2_batch += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_batch = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)

            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
                xx = torch.cat((xx[..., step:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_batch += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    train_loss[ep] = train_l2_step/ntrain/(T/step)
    test_loss[ep] = test_l2_step/ntest/(T/step)
    
    t2 = default_timer()
    scheduler.step()
    print('Epoch-{}, Time-{:0.4f}, Train-L2-Batch-{:0.4f}, Train-L2-Step-{:0.4f}, Test-L2-Batch-{:0.4f}, Test-L2-Step-{:0.4f}'
          .format(ep, t2-t1, train_l2_step/ntrain/(T/step), train_l2_batch/ntrain, test_l2_step/ntest/(T/step),
          test_l2_batch/ntest))

# %%
""" Prediction """
prediction = []
test_e = []     
with torch.no_grad():
    
    index = 0
    for xx, yy in test_loader:
        test_l2_step = 0
        test_l2_batch = 0
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            xx = torch.cat((xx[..., step:], im), dim=-1)
            
        prediction.append( pred.cpu() )
        test_l2_step += loss.item()
        test_l2_batch += myloss(pred.reshape(1, -1), yy.reshape(1, -1)).item()
        test_e.append( test_l2_step )
        index += 1
        
        print("Batch-{}, Test-loss-step-{:0.6f}, Test-loss-batch-{:0.6f}".format(
            index, test_l2_step/batch_size/(T/step), test_l2_batch) )
        
prediction = torch.cat((prediction))
test_e = torch.tensor((test_e))         
print('Mean Testing Error:', 100*torch.mean(test_e).numpy()/batch_size/(T/step), '%')

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
        plt.imshow(prediction[batch_no,:,:,tvalue], cmap='gist_ncar', interpolation='Gaussian')
        plt.title('Day-{}'.format(tvalue+1)) 
        plt.xlabel('Longitude ($^{\circ}$)', fontweight='bold');
        plt.grid(True)
        if index == 0 or index == 3:
            plt.ylabel('Prediction \n Latitude ($^{\circ}$)', fontweight='bold')
        else:
            plt.ylabel('Latitude ($^{\circ}$)', fontweight='bold')
        index = index + 1
        
# %%
"""
For saving the trained model and prediction data
"""
torch.save(model, 'model/model_wno_ERA5_time')
scipy.io.savemat('results/wno_results_ERA5_time.mat', mdict={'x_test':x_test.cpu().numpy(),
                                                    'y_test':y_test.cpu().numpy(),
                                                    'pred':prediction.cpu().numpy(),  
                                                    'test_e':test_e.cpu().numpy()})


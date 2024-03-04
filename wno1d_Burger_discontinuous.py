"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
   
-- This code is for 1-D Burgers' equation with discontinuous field (time-dependent problem).
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
        
        Input : (T_in+1)-channel tensor, first T_in step and location (u(x,t0),...u(x,t_T), x)
              : shape: (batchsize * x=s * c=T_in+1)
        Output: Solution of a later timestep (u(x, T_in+1))
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
            if index != self.layers - 1:
                x = convl(x) + wl(x) 
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

PATH = 'data/Burger_data/pde_burger/burgers_data_512_51.mat'
ntrain = 480
ntest = 20

batch_size = 20
learning_rate = 0.001

epochs = 500
step_size = 50   # weight-decay step size
gamma = 0.5      # weight-decay rate

wavelet = 'db6'  # wavelet basis function
level = 6        # lavel of wavelet decomposition
width = 40       # uplifting dimension
layers = 4       # no of wavelet layers

sub = 1          # subsampling rate
h = 512          # total grid size divided by the subsampling rate
grid_range = 1
in_channel = 21  # input channel is 21: (20 for a(x,t1-t20), 1 for x)

T_in = 20        # No of initial temporal-samples
T = 30           # No of prediction steps
step = 1         # Look-ahead step size

# %%
""" Read data """
dataloader = MatReader(PATH)
data = dataloader.read_field('sol') # N x Nx x Nt

x_train = data[:ntrain, ::sub, :T_in] 
y_train = data[:ntrain, ::sub, T_in:T_in+T] 

x_test = data[-ntest:, ::sub, :T_in] 
y_test = data[-ntest:, ::sub, T_in:T_in+T] 

x_train = x_train.reshape(ntrain,h,T_in)
x_test = x_test.reshape(ntest,h,T_in)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                          batch_size=batch_size, shuffle=False)

# %%
""" The model definition """
model = WNO1d(width=width, level=level, layers=layers, size=h, wavelet=wavelet,
              in_channel=in_channel, grid_range=grid_range).to(device)
print(count_params(model))

""" Training and testing """
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
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
            y = yy[..., t:t + step]
            im = model(xx)
            loss += myloss(im.reshape(batch_size, -1), y.reshape(batch_size, -1))
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            xx = torch.cat((xx[..., step:], im), dim=-1)
            
        train_l2_step += loss.item()
        l2_full = myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_batch += l2_full.item()

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
            loss += myloss(im.reshape(y.shape[0], -1), y.reshape(y.shape[0], -1))
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
            xx = torch.cat((xx[..., step:], im), dim=-1)
            
        prediction.append( pred )
        test_l2_step = loss.item()
        test_l2_batch = myloss(pred.reshape(1, -1), yy.reshape(1, -1)).item()
        
        test_e.append( test_l2_step/len(test_loader)/(T/step) )
        index += 1
        
        print("Batch-{}, Test-loss-step-{:0.6f}, Test-loss-batch-{:0.6f}".format(
            index, test_l2_step/len(test_loader)/(T/step), test_l2_batch/len(test_loader)) )
        
prediction = torch.cat((prediction))
test_e = torch.tensor((test_e))  
print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')

# %%
plt.rcParams['font.family'] = 'Times New Roman' 
plt.rcParams['font.size'] = 16
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

xtest = torch.cat([x_test,y_test.cpu()],axis=-1)
xpred = torch.cat([x_test,prediction.cpu()],axis=-1)

sample = 16
""" Solution and colocation points """
fig = plt.figure(figsize=(12, 5), dpi=100)
plt.subplots_adjust(wspace=0.3)
plt.subplot(1,2,1)
plt.imshow(xtest[sample, ...].cpu().numpy(), interpolation='nearest', cmap='rainbow', 
            extent=[0,1,-1,1], origin='lower', aspect='auto')
plt.colorbar(aspect=15, pad=0.015)
plt.title('Ground Truth', fontsize = 20) # font size doubled
plt.axvline(x=0.25, color='w', linewidth = 1)
plt.axvline(x=0.50, color='w', linewidth = 1)
plt.axvline(x=0.75, color='w', linewidth = 1)
plt.xlabel(r'$t$', size=12)
plt.ylabel(r'$x$', size=12)

plt.subplot(1,2,2)
plt.imshow(xpred[sample, ...].cpu().numpy(), interpolation='nearest', cmap='rainbow', 
            extent=[0,1,-1,1], origin='lower', aspect='auto')
plt.colorbar(aspect=15, pad=0.015)
plt.title('Prediction', fontsize = 20) # font size doubled
plt.axvline(x=0.25, color='w', linewidth = 1)
plt.axvline(x=0.50, color='w', linewidth = 1)
plt.axvline(x=0.75, color='w', linewidth = 1)
plt.xlabel(r'$t$', size=12)
plt.ylabel(r'$x$', size=12)

plt.show()

# %%
""" Solution at slices """
fig = plt.figure(figsize=(14, 5), dpi=100)
fig.subplots_adjust(wspace=0.4)
slices = [12,25,38]
x = torch.linspace(-1,1,h)

sample = 16
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.plot(x, xtest[sample,:,slices[i]], 'b-', linewidth = 2, label = 'Exact')       
    plt.plot(x, xpred[sample,:,slices[i]], 'r--', linewidth = 2, label = 'Prediction')
    plt.xlabel('$x$')
    plt.ylabel('$u(t,x)$')    
    plt.title('$t = {}$'.format(0.01*slices[i]*2), fontsize = 15)
    plt.axis('square')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.grid(True, alpha=0.25)
    if i == 1:
        plt.legend(frameon=False, ncol=2, bbox_to_anchor=(1,-0.15))

plt.show()

# %%
""" For saving the trained model and prediction data """

torch.save(model, 'model/WNO_burgers_time_dependent')
scipy.io.savemat('results/wno_results_burgers_time_dependent.mat', mdict={'x_test':x_test.cpu().numpy(),
                                                    'y_test':y_test.cpu().numpy(),
                                                    'prediction':prediction.cpu().numpy(),  
                                                    'test_e':test_e.cpu().numpy()})

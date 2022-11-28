"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
   
-- This code is for 1-D Burgers' equation with discontinuous field (time-dependent problem).
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
import matplotlib.gridspec as gridspec

torch.manual_seed(0)
np.random.seed(0)

# %%
""" Def: 1d Wavelet layer """
class WaveConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, level, dummy):
        super(WaveConv1d, self).__init__()

        """
        1D Wavelet layer. It does Wavelet Transform and Inverse Wavelet Transform.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.level = level
        self.wavelet = 'db6'
        
        self.dwt_ = DWT1D(wave=self.wavelet, J=self.level, mode='symmetric').to(dummy.device)
        self.mode_data, _ = self.dwt_(dummy)
        self.modes1 = self.mode_data.shape[-1]
        
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1))

    # Convolution
    def mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute single tree Discrete Wavelet coefficients using some wavelet 
        dwt = DWT1D(wave=self.wavelet, J=self.level, mode='symmetric').to(x.device)
        x_ft, x_coeff = dwt(x)
        
        # Multiply the final low pass and high pass coefficients
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[-1],  device=x.device)
        out_ft = self.mul1d(x_ft, self.weights1)
        x_coeff[-1] = self.mul1d(x_coeff[-1], self.weights1)
        
        # Reconstruct the signal
        idwt = IDWT1D(wave=self.wavelet, mode='symmetric').to(x.device)
        x = idwt((out_ft, x_coeff))
        return x

""" The forward operation """
class WNO1d(nn.Module):
    def __init__(self, width, level, dummy_data):
        super(WNO1d, self).__init__()

        """
        The WNO network. It contains 4 layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. 4 layers of the integral operators v(+1) = g(K(.) + W)(v).
            W is defined by self.w_; K is defined by self.conv_.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        input: the solutions at previous 39 time steps and location (a(t-39,x), ..., a(t-1,x), x)
        input shape: (batchsize, x=s, c=40)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.level = level
        self.width = width
        self.dummy_data = dummy_data
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(21, self.width) # input channel is 40: (39 for a(x), 1 for x)

        self.conv0 = WaveConv1d(self.width, self.width, self.level, self.dummy_data)
        self.conv1 = WaveConv1d(self.width, self.width, self.level, self.dummy_data)
        self.conv2 = WaveConv1d(self.width, self.width, self.level, self.dummy_data)
        self.conv3 = WaveConv1d(self.width, self.width, self.level, self.dummy_data)
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
        # The grid of the solution
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(-1, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


# %%
""" Model configurations """

ntrain = 480
ntest = 20
s = 512

batch_size = 20
learning_rate = 0.001

epochs = 500
scheduler_step = 50
scheduler_gamma = 0.5

level = 6 
width = 40

sub_x = 1
sub_y = 1
T_in = 20
T = 30
step = 1

# %%
""" Read data """
dataloader = MatReader('data/Burger_data/burgers_data_512_51.mat')
data = dataloader.read_field('sol') # N x Nx x Nt

x_train = data[:ntrain, ::sub_x, :T_in] 
y_train = data[:ntrain, ::sub_x, T_in:T_in+T] 

x_test = data[-ntest:, ::sub_x, :T_in] 
y_test = data[-ntest:, ::sub_x, T_in:T_in+T] 

x_train = x_train.reshape(ntrain,s,T_in)
x_test = x_test.reshape(ntest,s,T_in)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                          batch_size=batch_size, shuffle=False)

# %%
""" The model definition """
model = WNO1d(width, level, x_train[0:1].permute(0,2,1)).to(device)
print(count_params(model))

""" Training and testing """
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

train_loss = torch.zeros(epochs)
test_loss = torch.zeros(epochs)
myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2_step = 0
    train_l2_full = 0
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
        train_l2_full += l2_full.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    test_l2_step = 0
    test_l2_full = 0
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
            test_l2_full += myloss(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1)).item()

    train_loss[ep] = train_l2_step/ntrain/(T/step)
    test_loss[ep] = test_l2_step/ntest/(T/step)
    
    t2 = default_timer()
    scheduler.step()
    print(ep, t2 - t1, train_l2_step / ntrain / (T / step), train_l2_full / ntrain, test_l2_step / ntest / (T / step),
          test_l2_full / ntest)

# %%
""" Prediction """
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
        
        print(index, test_l2_step / (T/step), test_l2_full / (T/step), mse.cpu().numpy())
        index = index + 1

print('Mean Testing Error:', 100*torch.mean(test_e).numpy() /(T/step), '%')

# %%
""" Plotting """ 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 18

figure3 = plt.figure(figsize = (14,9))
gs = gridspec.GridSpec(3, 4)
plt.subplots_adjust(hspace=0.5,wspace=0.45)

xtest = torch.cat([x_test,y_test],axis=-1)
xpred = torch.cat([x_test,pred0],axis=-1)
val = 6
figure3.text(0.035,0.45,'\n Prediction', rotation=90, color='green', fontsize=20, fontweight='bold')
figure3.text(0.035,0.75,'\n Truth', rotation=90, color='red', fontsize=20, fontweight='bold')

ax1 = figure3.add_subplot(gs[0, :])
plt.imshow(xtest.cpu()[val, :,:], extent = [0,1,-1,1], cmap='gist_ncar', aspect='auto', origin='lower')
plt.xlabel('t', fontweight='bold'); plt.ylabel('x', fontweight='bold');
plt.axvline(0.20, linewidth=4, color='k')
plt.axvline(0.40, linewidth=4, color='k')
plt.axvline(0.60, linewidth=4, color='k')
plt.axvline(0.80, linewidth=4, color='k')

ax2 = figure3.add_subplot(gs[1, :])
plt.imshow(xpred.cpu()[val, :,:], extent = [0,1,-1,1], cmap='gist_ncar', aspect='auto', origin='lower')
plt.xlabel('t', fontweight='bold'); plt.ylabel('x', fontweight='bold'); 
plt.axvline(0.20, linewidth=4, color='k')
plt.axvline(0.40, linewidth=4, color='k')
plt.axvline(0.60, linewidth=4, color='k')
plt.axvline(0.80, linewidth=4, color='k')

ax3 = figure3.add_subplot(gs[2, 0])
plt.plot(xtest.cpu()[val, :,10], linewidth=4)
plt.plot(xpred.cpu()[val, :,10], '--', linewidth=4)
plt.title('t = 0.2s', color='b', fontsize=20)
plt.xlabel('x', fontweight='bold'); plt.ylabel('u(x,t)', fontweight='bold'); 
plt.margins(0)

ax4 = figure3.add_subplot(gs[2, 1])
plt.plot(xtest.cpu()[val, :,20], linewidth=4)
plt.plot(xpred.cpu()[val, :,20], '--', linewidth=4)
plt.title('t = 0.4s', color='b', fontsize=20)
plt.xlabel('x', fontweight='bold'); plt.ylabel('u(x,t)', fontweight='bold'); 
plt.margins(0)

ax5 = figure3.add_subplot(gs[2, 2])
plt.plot(xtest.cpu()[val, :,30], linewidth=4)
plt.plot(xpred.cpu()[val, :,30], '--', linewidth=4)
plt.title('t = 0.6s', color='b', fontsize=20)
plt.xlabel('x', fontweight='bold'); plt.ylabel('u(x,t)', fontweight='bold'); 
plt.margins(0)

ax6 = figure3.add_subplot(gs[2, 3])
plt.plot(xtest.cpu()[val, :,40], linewidth=4)
plt.plot(xpred.cpu()[val, :,40], '--', linewidth=4)
plt.title('t = 0.8s', color='b', fontsize=20)
plt.xlabel('x', fontweight='bold'); plt.ylabel('u(x,t)', fontweight='bold'); 
plt.margins(0)

plt.legend(['Truth','Prediction'], ncol=2, bbox_to_anchor=(0.5, -0.3))


# %%
"""
For saving the trained model and prediction data
"""
# torch.save(model, 'model/model_wno_burger_discontinuous_time')
# scipy.io.savemat('pred/pred_wno_burger_discontinuous_time.mat', mdict={'pred': pred.cpu().numpy()})
# scipy.io.savemat('loss/train_loss_wno_burger_discontinuous_time.mat', mdict={'train_loss': train_loss.cpu().numpy()})
# scipy.io.savemat('loss/test_loss_wno_burger_discontinuous_time.mat', mdict={'test_loss': test_loss.cpu().numpy()})

# torch.cuda.empty_cache()

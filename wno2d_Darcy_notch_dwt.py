"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet Neural Operator for solving 
   parametric partialdifferential equations in computational mechanics problems.
   
-- This code is for 2-D Darcy equation in triangular domain with notch (time-independent problem).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

from timeit import default_timer
from utils import *
from wavelet_convolution import WaveConv2d

torch.manual_seed(0)
np.random.seed(0)

# %%
""" The forward operation """
class WNO2d(nn.Module):
    def __init__(self, width, level, layers, size, wavelet, in_channel, grid_range, padding=0):
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
        wavelet: string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: list with 2 elements (for 2D), right supports of 2D domain
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
        
        self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 3: (a(x, y), x, y)
        for i in range( self.layers ):
            self.conv.append( WaveConv2d(self.width, self.width, self.level, self.size, self.wavelet) )
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
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


# %%
""" Model configurations """

PATH = 'data/Darcy_Triangular_FNO.mat'
ntrain = 1000
ntest = 100

batch_size = 20
learning_rate = 0.001

epochs = 50
step_size = 50   # weight-decay step size
gamma = 0.5      # weight-decay rate

wavelet = 'db6'  # wavelet basis function
level = 3        # lavel of wavelet decomposition
width = 64       # uplifting dimension
layers = 4       # no of wavelet layers

sub = 2          # subsampling rate
h = int(((101 - 1)/sub) + 1) # total grid size divided by the subsampling rate
grid_range = [1, 1]          # The grid boundary in x and y direction
in_channel = 3   # (a(x, y), x, y) for this case

# %%
""" Read data """
reader = MatReader(PATH)
x_train = reader.read_field('boundCoeff')[:ntrain,::sub,::sub][:,:h,:h]
y_train = reader.read_field('sol')[:ntrain,::sub,::sub][:,:h,:h]

x_test = reader.read_field('boundCoeff')[-ntest:,::sub,::sub][:,:h,:h]
y_test = reader.read_field('sol')[-ntest:,::sub,::sub][:,:h,:h]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain,h,h,1)
x_test = x_test.reshape(ntest,h,h,1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train),
                                           batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test),
                                          batch_size=batch_size, shuffle=False)

# %%
""" The model definition """
model = WNO2d(width=width, level=level, layers=layers, size=[h,h], wavelet=wavelet,
              in_channel=in_channel, grid_range=grid_range, padding=1).to(device)
print(count_params(model))

""" Training and testing """
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_loss = torch.zeros(epochs)
test_loss = torch.zeros(epochs)
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
        out = model(x).reshape(batch_size, h, h)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)
        
        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
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

            out = model(x).reshape(batch_size, h, h)
            out = y_normalizer.decode(out)

            test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

    train_mse /= len(train_loader)
    train_l2/= ntrain
    test_l2 /= ntest
    
    train_loss[ep] = train_l2
    test_loss[ep] = test_l2
    
    t2 = default_timer()
    print("Epoch-{}, Time-{:0.4f}, Train-MSE-{:0.4f}, Train-L2-{:0.4f}, Test-L2-{:0.4f}"
          .format(ep, t2-t1, train_mse, train_l2, test_l2))
    
# %%
""" Prediction """
pred = []
test_e = []
with torch.no_grad():
    
    index = 0
    for x, y in test_loader:
        test_l2 = 0
        x, y = x.to(device), y.to(device)

        out = model(x).reshape(batch_size, h, h)
        out = y_normalizer.decode(out)
        pred.append( out )

        test_l2 += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()
        test_e.append( test_l2/batch_size )
        
        print("Batch-{}, Loss-{}".format(index, test_l2/batch_size) )
        index += 1

pred = torch.cat((pred))
test_e = torch.tensor((test_e))  
print('Mean Testing Error:', 100*torch.mean(test_e).numpy(), '%')

# %%
""" Plotting """ 
s = 1
xmax = s
ymax = s-8/51
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 14

figure1 = plt.figure(figsize = (18, 14))
figure1.text(0.04,0.17,'\n Error', rotation=90, color='purple', fontsize=20)
figure1.text(0.04,0.34,'\n Prediction', rotation=90, color='green', fontsize=20)
figure1.text(0.04,0.57,'\n Truth', rotation=90, color='red', fontsize=20)
figure1.text(0.04,0.75,'Boundary \n Condition', rotation=90, color='b', fontsize=20)
plt.subplots_adjust(wspace=0.7)
index = 0
for value in range(y_test.shape[0]):
    if value % 29 == 1:
        plt.subplot(4,4, index+1)
        plt.title('B.C.-{}'.format(index+1), color='b', fontsize=18, fontweight='bold'); 
        plt.imshow(x_test[value,:,:,0], cmap='nipy_spectral', extent=[0,1,0,1], origin='lower', interpolation='Gaussian')
        plt.xlabel('x',fontweight='bold'); plt.ylabel('y',fontweight='bold')
        plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
        
        xf = np.array([0., xmax/2]); yf = xf*(ymax/(xmax/2)); plt.fill_between(xf, yf, ymax, color = [1, 1, 1])
        xf = np.array([xmax/2, xmax]); yf = (xf-xmax)*(ymax/((xmax/2)-xmax)); plt.fill_between(xf, yf, ymax, color = [1, 1, 1])
        xf = np.array([0, xmax]); plt.fill_between(xf, ymax, s, color = [1, 1, 1])        
        plt.gca().add_patch(Rectangle((0.5,0),0.01,0.41, facecolor='white'))
        
        ###
        plt.subplot(4,4, index+1+4)
        plt.imshow(y_test[value,:,:], origin='lower', extent = [0, 1, 0, 1], interpolation='Gaussian', cmap='nipy_spectral')
        plt.colorbar(fraction=0.045)
        plt.xlabel('x',fontweight='bold'); plt.ylabel('y',fontweight='bold')
        plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
        
        xf = np.array([0., xmax/2]); yf = xf*(ymax/(xmax/2)); plt.fill_between(xf, yf, ymax, color = [1, 1, 1])
        xf = np.array([xmax/2, xmax]); yf = (xf-xmax)*(ymax/((xmax/2)-xmax)); plt.fill_between(xf, yf, ymax, color = [1, 1, 1])
        xf = np.array([0, xmax]); plt.fill_between(xf, ymax, s, color = [1, 1, 1])        
        plt.gca().add_patch(Rectangle((0.5,0),0.01,0.41, facecolor='white'))
        
        ###
        plt.subplot(4,4, index+1+8)
        plt.imshow(pred[value,:,:].cpu(), origin='lower', extent = [0, 1, 0, 1], interpolation='Gaussian', cmap='nipy_spectral')
        plt.colorbar(fraction=0.045)
        plt.xlabel('x',fontweight='bold'); plt.ylabel('y',fontweight='bold')
        plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
        
        xf = np.array([0., xmax/2]); yf = xf*(ymax/(xmax/2)); plt.fill_between(xf, yf, ymax, color = [1, 1, 1])
        xf = np.array([xmax/2, xmax]); yf = (xf-xmax)*(ymax/((xmax/2)-xmax)); plt.fill_between(xf, yf, ymax, color = [1, 1, 1])
        xf = np.array([0, xmax]); plt.fill_between(xf, ymax, s, color = [1, 1, 1])        
        plt.gca().add_patch(Rectangle((0.5,0),0.01,0.4, facecolor='white'))
        
        ###
        plt.subplot(4,4, index+1+12)
        plt.imshow(np.abs(y_test[index,:,:]-pred[index,:,:].cpu()), cmap='jet', extent=[0,1,0,1], interpolation='Gaussian', origin='lower')
        plt.colorbar(fraction=0.045)
        plt.xlabel('x',fontweight='bold'); plt.ylabel('y',fontweight='bold')
        plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');

        xf = np.array([0., xmax/2]); yf = xf*(ymax/(xmax/2)); plt.fill_between(xf, yf, ymax, color = [1, 1, 1])
        xf = np.array([xmax/2, xmax]); yf = (xf-xmax)*(ymax/((xmax/2)-xmax)); plt.fill_between(xf, yf, ymax, color = [1, 1, 1])
        xf = np.array([0, xmax]); plt.fill_between(xf, ymax, s, color = [1, 1, 1])        
        plt.gca().add_patch(Rectangle((0.5,0),0.01,0.41, facecolor='white'))
        plt.margins(0)
        
        index = index + 1

# %%
""" For saving the trained model and prediction data """
torch.save(model, 'model/WNO_darcy_notch')
scipy.io.savemat('results/wno_results_darcy_notch.mat', mdict={'x_test':x_test.cpu().numpy(),
                                                    'y_test':y_test.cpu().numpy(),
                                                    'pred':pred.cpu().numpy(),  
                                                    'test_e':test_e.cpu().numpy()})

# -*- coding: utf-8 -*-

# Example figure 1

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
np.random.seed(0)
torch.set_printoptions(sci_mode=False)

def tilted_loss(y,f,q):
    e = -(y-f)
    return torch.mean(torch.max(q*e, (q-1)*e))


class nn_simple(nn.Module):
    def __init__(self, input_size=1, output_size=1, nhidden=64, bias=True, database_input=None, database_output=None):
        super(nn_simple, self).__init__()
        self.linear1 = nn.Linear(input_size,nhidden,bias=bias)
        self.linear2 = nn.Linear(nhidden,nhidden,bias=bias)
        self.linear3 = nn.Linear(nhidden,output_size,bias=bias)
        self.act = nn.ELU(inplace=False)
        if database_input is not None and database_output is not None:
            self.mean_input = torch.mean(database_input,axis=0)
            self.std_input = torch.std(database_input,axis=0)
            self.mean_output = torch.mean(database_output,axis=0)
            self.std_output = torch.std(database_output,axis=0)
        else:
            self.mean_input = 0.0
            self.std_input = 1.0
            self.mean_output = 0.0
            self.std_output = 1.0

    def forward(self, x, q):
        x = (x-self.mean_input) / self.std_input

        x = self.act(self.linear1(x))
        x += q #QUNTILE INFO
        x = self.act(self.linear2(x))
        x = self.linear3(x)
        return x*self.std_output + self.mean_output




# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Sampling a spectral line:
n_profiles = 10000
n_lambda = 101
noise = 1e-4
wvl = np.linspace(-10,10,n_lambda)
v = np.random.normal(-0.0, 0.4, size=n_profiles)
dv = np.random.uniform(low=1.0, high=5.5, size=n_profiles)
amp = np.random.uniform(low=0.9, high=0.2, size=n_profiles)

# test_profile
dv[2], amp[2], v[2] = 3.4, 0.548, 0.49
test_profile = 1.0 - amp[2]*np.exp(-(wvl[:]+v[2])**2 / dv[2]**2)

stokes = 1.0 - amp[:,None]*np.exp(-(wvl[None,:]+v[:,None])**2 / dv[:,None]**2)
# stokes += np.random.normal(loc=0.0, scale=noise, size=stokes.shape) # Later during training
stokes = np.expand_dims(stokes, axis=1)


# Check a small set of samples:
doplot = False
if doplot == True:
    plt.figure()
    for i in range(15):
        plt.plot(stokes[i,0,:])
    plt.savefig('stokes_sample_.pdf')

print('stokes.shape: ',stokes.shape)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

npoints = 5
ni_epochs = 5000
diff_point = []
epsilon = 1e-8
lrnet = 1e-3
lrnet2 = 1e-5


zoom = 1.0#2
fig2, ax = plt.subplots(4,1,figsize=(4*zoom,14*zoom),sharex=True)

x = np.array([0.0, stokes.shape[2]-1])
newpoints2add = npoints - 2
for jj in range(newpoints2add+1):

    xx = x.astype('int')#.astype('float32')

    input_size = len(xx)
    output_size = stokes.shape[2]


    x_torch = torch.from_numpy(stokes[:,:,xx].astype('float32'))
    y_torch = torch.from_numpy(stokes[:,:,:].astype('float32'))

    loss_fn = tilted_loss
    mod = nn_simple(input_size,output_size,database_input=x_torch,database_output=y_torch) #Our model 

    qs = [0.16, 0.5, 0.84]


    optimizer = torch.optim.Adam(mod.parameters(), lr=lrnet)
    for loop in tqdm(range(ni_epochs+1)):
        qq = np.random.choice(qs) #Random value to learn other possible values
        qq = torch.from_numpy(np.array(qq))

        optimizer.zero_grad()        #reset gradients
        out = mod(x_torch,qq)           #evaluate model

        noise_temp = torch.from_numpy(np.random.normal(loc=0.0, scale=noise, size=stokes.shape).astype('float32'))

        loss = loss_fn(out, y_torch+noise_temp, qq) #calculate loss
        loss.backward()              #calculate gradients
        optimizer.step()             #step fordward

    
    # Second training step with small lr to improve estimation
    optimizer = torch.optim.Adam(mod.parameters(), lr=lrnet2)
    for loop in tqdm(range(ni_epochs+1)):
        qq = np.random.choice(qs) #Random value to learn other possible values
        qq = torch.from_numpy(np.array(qq))

        optimizer.zero_grad()        #reset gradients
        out = mod(x_torch,qq)           #evaluate model

        noise_temp = torch.from_numpy(np.random.normal(loc=0.0, scale=noise, size=stokes.shape).astype('float32'))

        loss = loss_fn(out, y_torch+noise_temp, qq) #calculate loss
        loss.backward()              #calculate gradients
        optimizer.step()             #step fordward


    # TEST PROFILE
    x_torch = torch.from_numpy(stokes[2:2+1,:,xx].astype('float32'))

    out = mod(x_torch,qs[1])
    out_16 = mod(x_torch,qs[0])
    out_84 = mod(x_torch,qs[2])
    diff = torch.mean((out_84 - out_16)**2.,axis=0)[0,:]
    newpoint = torch.argmax(diff)
    print('n= '+str(len(xx))+' points:', xx , ' -> newpoint: ', newpoint.item(), ' {0:.2e}'.format(loss.item()) )#, end='\r')

    out_test = out.detach().numpy()
    out_16 = out_16.detach().numpy()
    out_84 = out_84.detach().numpy()


    # Some plots
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    ax[jj].set_title('NN, Iteration '+str(jj+1),fontsize=10)
    ax[jj].plot(wvl,test_profile,label='target',color='C0')
    ax[jj].scatter(wvl[x.astype('int')],test_profile[x[:].astype('int')], s=30, marker='o',color='C3',label='sampling')
    ax[jj].plot(wvl,out_test[0,0,:],label='prediction',color='C1')# $\pm\ \sigma$
    ax[jj].fill_between(wvl,out_16[0,0,:],out_84[0,0,:], alpha=0.5,color='C1')
    # ax[jj].set_ylabel('Intensity [a.u., n='+str(2+jj)+' points]')
    ax[jj].yaxis.set_ticklabels([])
    ax[jj].set_ylim([0.2,1.3])
    ax[jj].legend(loc='lower right')
    ax[jj].minorticks_on()


    if jj < newpoints2add:
        x = np.append(x, newpoint)
    # print(x)

ax[jj].set_xlabel('Wavelength axis [a.u.]')
plt.savefig('im_sampling_gaussian_nn.pdf')
plt.close(fig2)

# Save the final sampling
# np.save('output_sampling.npy',xx)
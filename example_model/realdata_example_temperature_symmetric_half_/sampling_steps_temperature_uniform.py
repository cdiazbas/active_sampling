import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy import interpolate
from scipy import optimize
from tqdm import tqdm
torch.set_printoptions(sci_mode=False)
from resnet_model import ResidualNet

"""
Testing uniform sampling for the CaII line at 8542A using the temperature
Coded by Carlos Diaz (UiO-RoCS, 2022)
"""

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Sampling a spectral line:
stokes = np.load('output/stokes.npy')
stokes = np.expand_dims(stokes, axis=1)

stokes = np.concatenate((stokes,stokes[:,:,::-1])) # make that symmetric!
stokes = stokes[:,:,16:-15] # make that symmetric!
stokes = stokes[:,:,:121]#[:,:,::2]

temp = np.load('output/temperature.npy')/1e3
temp = np.expand_dims(temp, axis=1)
temp = np.concatenate((temp,temp)) # make that symmetric!


for i in range(15):
    plt.plot(stokes[i,0,:])
plt.minorticks_on()
plt.ylabel('Intensity axis [au]')
plt.xlabel('Wavelength axis [index]')
plt.savefig('output/stokes_sample_.pdf')

print(stokes.shape)

import os
if not os.path.exists('output/sampling_uniform'):
   os.makedirs('output/sampling_uniform')



# STEPS:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

results_xx = []
results_chi2 = []


def f_nn(x,edges=None,ni_epochs=5000):
    
    chi2 = 0.0
    lrstep = 6
    x = np.append(x, edges)
    xx = x.astype('int')#.astype('float32')


    input_size = len(xx)
    output_size = temp.shape[2]


    x_torch = torch.from_numpy(stokes[:,:,xx].astype('float32'))
    y_torch = torch.from_numpy(temp[:,:,:].astype('float32'))

    mod = ResidualNet(in_features=input_size,out_features=output_size,database_input=x_torch,database_output=y_torch,hidden_features=64,num_blocks=5) #Our model 
    
    loss_array = []
    optimizer = torch.optim.Adam(mod.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(ni_epochs/lrstep), gamma=0.4)

    for loop in tqdm(range(ni_epochs+1),leave=True):
        optimizer.zero_grad()        #reset gradients
        out = mod(x_torch)           #evaluate model
        
        diff = torch.mean((out[:,0,:] - y_torch[:,0,:])**2.,axis=0)
        weight = torch.ones_like(diff)
        loss = torch.mean(diff*weight)
        
        loss.backward()              #calculate gradients
        optimizer.step()             #step fordward
        
        loss_array.append(loss.item())

        scheduler.step()

    chi2 = loss.item()

    # Neural network training:
    fig1 = plt.figure()
    plt.plot(loss_array)
    plt.title('loss_final: {0:2.2e}'.format( np.min(loss_array)))
    plt.yscale('log')
    plt.minorticks_on()
    plt.savefig('output/sampling_uniform/error_'+str(inpoint)+'.pdf')
    plt.close(fig1)

    fig1 = plt.figure()
    plt.plot(diff.detach().numpy(),label='MSE')
    # plt.plot(weight.detach().numpy(),label='Weight')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Logtau axis [index]')
    plt.ylabel('Mean squared error')
    plt.savefig('output/sampling_uniform/stokes_error_temp'+str(input_size)+'.png')
    plt.close(fig1)

    np.save('output/sampling_uniform/stokes_error_temp'+str(input_size)+'.npy',diff.detach().numpy())

    print('n= '+str(len(xx))+' points ->', xx, '{0:.2e}'.format(loss.item()) )

    results_xx.append(xx[0])
    results_chi2.append(chi2)

    return chi2



# STEPS:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
npoints = 9
ni_epochs = 10000

for inpoint in range(3,npoints+1):

    x = np.linspace(0.0, stokes.shape[2]-1,inpoint).astype('int')
    print('=>',x)
    xfun = f_nn(x[0],edges=x[1:],ni_epochs=ni_epochs)


    f, ax = plt.subplots(1, 5, sharey=True,figsize=(20,5))
    for ii in range(5):
        ax[ii].plot(stokes[ii,0,:],label='target')
        # ax[ii].plot(out[ii,0,:],label='output')
        ax[ii].scatter(x[:],stokes[ii,0,x[:].astype('int')],color='red',label='sampling')

    plt.legend()
    plt.xlabel('Wavelength axis [index]')
    plt.ylabel('Intensity [n='+str(len(x))+' points]')
    plt.savefig('output/sampling_uniform/stokes_reconstruction_temp_'+str(len(x))+'.png')


# Save final sampling
np.save('output/sampling_uniform/final_sampling.npy',x)

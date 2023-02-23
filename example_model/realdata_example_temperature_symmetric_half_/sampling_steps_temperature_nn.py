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
Finding best sampling for the CaII line at 8542A using the temperature
Coded by Carlos Diaz (UiO-RoCS, 2022)
"""


# Sampling a spectral line:
stokes = np.load('stokes.npy')
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


# STEPS:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

x = np.array([0.0,stokes.shape[2]-1])
npoints = 9
ni_epochs = 10000
lrstep = 6
avoid = 3 # it is possible to avoid evaluate close points
dirname = 'output/sampling_nn/'


diff_point = []
for inpoint in range(3,npoints+1):

    availablepoints_ = [i for i in np.arange(stokes.shape[-1]) if i not in x]

    # After removing close points
    availablepoints = [i for i in availablepoints_ if np.min(np.abs(x-i))> avoid]


    x_torch = torch.from_numpy(stokes[:,:,x.astype('int')].astype('float32'))
    y_torch = torch.from_numpy(temp[:,:,:].astype('float32'))

    input_size = len(x)
    output_size = temp.shape[-1]

    qq = np.random.choice(availablepoints,stokes.shape[0])
    database_context = torch.from_numpy(np.stack([stokes[range(len(qq)),0,qq],qq]).T[:,:].astype('float32'))

    mod = ResidualNet(in_features=input_size,out_features=output_size,database_input=x_torch,database_output=y_torch,hidden_features=64,num_blocks=5, context_features=2,database_context=database_context) #Our model 


    loss_array = []
    optimizer = torch.optim.Adam(mod.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(ni_epochs/lrstep), gamma=0.4)

    for loop in tqdm(range(ni_epochs+1),leave=True):
        optimizer.zero_grad()        #reset gradients

        qq = np.random.choice(availablepoints,stokes.shape[0])
        context = torch.from_numpy(np.stack([stokes[range(len(qq)),0,qq],qq]).T[:,:].astype('float32'))
        out = mod(x_torch[:,0,:], context=context)           #evaluate model
        
        diff = torch.mean((out[:,:] - (y_torch[:,0,:]))**2.,axis=0)
        weight = torch.ones_like(diff)
        loss = torch.mean(diff*weight)
        
        loss.backward()              #calculate gradients
        optimizer.step()             #step fordward

        loss_array.append(loss.item())

        scheduler.step()



    # Testing:
    qresults_chi2 = []
    qresults_diff = []
    for ii in range(len(availablepoints)):
        qq = availablepoints[ii]
        context = torch.from_numpy(np.stack([stokes[:,0,qq],np.repeat(qq, stokes.shape[0], axis=0)]).T[:,:].astype('float32'))
        out = mod(x_torch[:,0,:], context=context)           #evaluate model
        diff = torch.mean((out[:,:] - (y_torch[:,0,:]))**2.,axis=0)
        loss = torch.mean(diff)
        
        qresults_chi2.append(loss.item())
        qresults_diff.append(diff.detach().numpy())

    newindex = np.argmin(qresults_chi2)
    newpoint = availablepoints[newindex]


    diff_point.append([qresults_chi2,newpoint,inpoint,availablepoints,newindex])


    # Neural network training:
    fig1 = plt.figure()
    plt.plot(loss_array)
    plt.title('loss_final: {0:2.2e}'.format( np.min(loss_array)))
    plt.yscale('log')
    plt.minorticks_on()
    plt.savefig(dirname+'error_'+str(inpoint)+'.pdf')
    plt.close(fig1)

    fig1 = plt.figure()
    for qq in range(len(diff_point)):
        plt.plot(diff_point[qq][3],diff_point[qq][0],label='it= '+str(diff_point[qq][2]),zorder=qq-0.5)
        plt.scatter(diff_point[qq][1],diff_point[qq][0][diff_point[qq][4]],zorder=qq)
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Wavelength axis [index]')
    plt.ylabel('Mean squared error - Temperature [kK]')
    plt.minorticks_on()
    plt.savefig(dirname+'comb_stokes_error_temp'+str(inpoint)+'.png')
    plt.close(fig1)


    fig1 = plt.figure()
    plt.plot(availablepoints, qresults_chi2,'.-',zorder=-0.5,color='k')
    plt.scatter(newpoint,qresults_chi2[newindex],zorder=0.0)
    plt.yscale('log')
    plt.xlabel('Wavelength axis [index]')
    plt.ylabel('Mean squared error - Temperature [kK]')
    plt.minorticks_on()
    plt.savefig(dirname+'im_stokes_error_temp'+str(inpoint)+'.png')
    plt.close(fig1)




    print('n= '+str(len(x))+' +1 points ->',x.astype(np.int), newpoint, '=> {0:.2e}'.format(np.max(qresults_chi2)) )#, end='\r')

    np.save(dirname+'stokes_error_temp'+str(inpoint)+'.npy',qresults_diff[newindex])
    x = np.append(x,newpoint)




    f, ax = plt.subplots(1, 5, sharey=True,figsize=(20,5))
    for ii in range(5):
        ax[ii].plot(stokes[ii,0,:],label='target')
        # ax[ii].plot(out[ii,0,:],label='output')
        ax[ii].scatter(x[:],stokes[ii,0,x[:].astype('int')],color='red',label='sampling')

    plt.legend()
    plt.xlabel('Wavelength axis [index]')
    plt.ylabel('Intensity [n='+str(len(x))+' points]')
    plt.savefig(dirname+'stokes_reconstruction_temp_'+str(inpoint)+'.png')


# Save final sampling
np.save(dirname+'final_sampling.npy',x)

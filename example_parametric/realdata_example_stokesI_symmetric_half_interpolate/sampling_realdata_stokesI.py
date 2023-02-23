# -*- coding: utf-8 -*-
# Using MinError + neural network

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
torch.set_printoptions(sci_mode=False)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def sampling_mode(size, npoints, ninner, distance, multiple=1):

    innerside = np.arange(ninner)*distance
    outterside = np.arange(npoints-ninner+1)*distance*multiple
    output = np.concatenate((size -innerside[:-1],size -(outterside+innerside[-1])), axis=0).astype('float32')[::-1]
    # output
    # print((-output[1:]+output[0:-1]))
    if np.abs(np.max(output)-np.min(output)) > size:
        return None
    else:
        return  output.copy()
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


# Training set
stokes = np.load('stokes.npy')
# print(stokes.shape)

stokes = np.expand_dims(stokes, axis=1)
stokes = np.concatenate((stokes,stokes[:,:,::-1])) # make that symmetric!
stokes = stokes[:,:,16:-15] # make that symmetric!
stokes = stokes[:,:,:121]#[:,:,::2]
print('stokes.shape',stokes.shape)
stokes = stokes[:1000,:,:]

# SAMPLING DE LA CRUZ 2012
ca8_idxs = np.array([0,20,40,46,48,50,52,54,56,58,60])*2
# print(len(ca8_idxs))

wav = np.load('wav.npy')[16:-15][:121]#[::2]
wav -= wav[-1] # Centrered
print('wav.shape',wav.shape)
# print('wav',wav)
print('wav comparison:',np.around(sorted(wav[ca8_idxs.astype('int')]),3))

plt.figure()
for i in range(15):
    plt.plot(wav,stokes[i,0,:])
plt.savefig('stokes_sample_.pdf')
print(stokes.shape)


# [7,4//9,1]
npoints = 7
print('Total points',(npoints-1)*2 +1)
ni_epochs = 10000#00#//10
diff_point = []
lrnet = 5e-4
epsilon = 1e-7
lrstep = 6
noise = 1e-4
batch_size = 1000
# mm = 10 # Distance factor after inner points

input_size = npoints
output_size = stokes.shape[2]
y_torch = torch.from_numpy(stokes[:,:,:].astype('float32'))
loss_fn = nn.MSELoss()

# for mm in range(1,10):
for mm in [5]:

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



    # # Some plots of progress
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print('Final plots ...')
    maxi = (output_size-1)/(npoints-1)
    darray = np.linspace(0.1,maxi,40).astype('float32')
    dinner = np.arange(1.0,npoints+1).astype('float32')
    losstest_ = np.zeros((len(darray),len(dinner)))*np.nan
    paramtest = []
    losstest = []
    from tqdm import tqdm
    for ii in tqdm(range(darray.size)):
        for jj in tqdm(range(dinner.size),leave=False):
            sampling = sampling_mode(output_size-1, npoints, dinner[jj], darray[ii], multiple=mm)
            # print(sampling)
            if sampling is not None:
                dparam = torch.from_numpy(np.concatenate((darray[ii,None],dinner[jj,None]), axis=0))
                xnew = torch.from_numpy(sampling)

                
                from interp_1d import LinearInterp1D
                # Interpolation forward:
                yq_interpol = LinearInterp1D(torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1), y_torch[:,0,:])
                yq_cpu = yq_interpol._interp(xnew,None)

                # Interpolation back:
                # from interp_1d import CubicSpline1D
                # yq_interpol = CubicSpline1D(xnew.repeat(stokes.shape[0],1), yq_cpu,bc_type="natural")

                # # yq_interpol = LinearInterp1D(xnew.repeat(stokes.shape[0],1), yq_cpu)
                # yq_back = yq_interpol._interp(torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1),None)
                from weno4 import weno4
                yq_back = torch.from_numpy(np.array(list(map(weno4, torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1), xnew.repeat(stokes.shape[0],1),yq_cpu))))

                

                lossitem = loss_fn(yq_back, y_torch[:,0,:]).item()
                losstest.append(lossitem)
                paramtest.append((darray[ii],dinner[jj]))
                losstest_[ii,jj] = lossitem
            else:
                pass


    # Best combination:
    fig1 = plt.figure(figsize=(4,6))
    deltawl = (wav[1]-wav[0])*1e3
    plt.axhline(  deltawl*paramtest[np.nanargmin(losstest)][0]  ,color='black',ls='--')
    plt.axvline(  paramtest[np.nanargmin(losstest)][1]  ,color='black',ls='--')
    plt.imshow(losstest_[1:,:],norm=colors.LogNorm(),cmap='magma_r',extent=[-0.5+dinner[0],dinner[-1]+0.5,deltawl*darray[1],deltawl*darray[-1]],origin='lower' , aspect='auto')
    plt.title(r'Best: d={0:2.0f} $\rm m\AA$, inner={1:2.0f}, mse={2:1.0e}, outdf={3:1.0f}'.format( deltawl*paramtest[np.nanargmin(losstest)][0], paramtest[np.nanargmin(losstest)][1]  ,np.nanmin(losstest),mm),size=11)
    plt.minorticks_on()
    # plt.ylim(deltawl*darray[1],deltawl*darray[-1])
    plt.xlabel('Inner points')
    plt.ylabel(r'Distance [$\rm m\AA$]')
    plt.savefig('im_nn_'+str(mm)+'.pdf')
    plt.close(fig1)





    # Final sampling:
    x = sampling_mode(output_size-1, npoints, paramtest[np.nanargmin(losstest)][1], paramtest[np.nanargmin(losstest)][0], multiple=mm)
    nwav = np.interp(x, np.arange(wav.size), wav)

    print('final_sampling [interal units]: ',x)
    print('final_sampling [wavelength]: ',nwav)
    np.save('final_sampling.npy',x)





    xnew = torch.from_numpy(x)

    from interp_1d import LinearInterp1D
    # Interpolation forward:
    yq_interpol = LinearInterp1D(torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1), y_torch[:,0,:])
    yq_cpu = yq_interpol._interp(xnew,None)

    # Interpolation back:
    # from interp_1d import CubicSpline1D
    # yq_interpol2 = CubicSpline1D(xnew.repeat(stokes.shape[0],1), yq_cpu,bc_type="natural")
    # # yq_interpol2 = LinearInterp1D(xnew.repeat(stokes.shape[0],1), yq_cpu)
    # yq_back = yq_interpol2._interp(torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1),None)
    from weno4 import weno4
    yq_back = torch.from_numpy(np.array(list(map(weno4, torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1), xnew.repeat(stokes.shape[0],1),yq_cpu))))


    # VERY BAD!!
    # from interp1d import interp1d
    # out1 = interp1d(torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1), y_torch[:,0,:], xnew)
    # out = interp1d(xnew, out1, torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1))


    out_test = yq_back.detach().numpy()
    zoom = 0.8
    fig2, ax = plt.subplots(1, 5, sharey=True,figsize=(20*zoom,4.5*zoom))
    liseg = np.array([16,18,15,21,20])
    for ii in range(len(ax)):
        # ax[ii].set_title(str(liseg[ii]))
        ax[ii].plot(wav,stokes[liseg[ii],0,:],label='target')
        ax[ii].plot(wav,out_test[liseg[ii],:],label='output')
        ax[ii].scatter(nwav,yq_cpu[liseg[ii],:],color='red',label='scheme',zorder=2,s=30.0,alpha=0.8)
        ax[ii].minorticks_on()

        for kk in range(len(nwav)):
            ax[ii].axvline(nwav[kk],ls='-',alpha=0.1,color='k')
    ax[0].legend()
    ax[0].set_xlabel(r"$\lambda - 8542.1$ $[\rm \AA]$")
    ax[0].set_ylabel(r'Stokes I/I$\rm _C$')
    plt.locator_params(axis='y', nbins=5)
    plt.savefig('im_stokes_'+str(mm)+'.pdf',bbox_inches='tight')
    plt.close(fig2)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




# -*- coding: utf-8 -*-
# Using MaxError + neural network

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
torch.set_printoptions(sci_mode=False)


# Training set
stokes = np.load('stokes.npy')
print(stokes.shape)

stokes = np.expand_dims(stokes, axis=1)
stokes = np.concatenate((stokes,stokes[:,:,::-1])) # make that symmetric!
stokes = stokes[:,:,16:-15] # make that symmetric!
stokes = stokes[:,:,:121]#[:,:,::2]
print(stokes.shape)

npoints = 9
dparam = 15.0
xnew = torch.arange(npoints)*dparam
xnew = 120-xnew
print(xnew)
xnew, indices = torch.sort(xnew)
print(xnew)

stokes = torch.from_numpy(stokes[:,:,:].astype('float32'))
# from interp1d import interp1d
# yq_cpu = interp1d(torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1), stokes[:,0,:], xnew, None)

from interp_1d import LinearInterp1D
yq_interpol = LinearInterp1D(torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1), stokes[:,0,:])
yq_cpu = yq_interpol._interp(xnew,None)


# from interp1d import interp1d
# out = interp1d(torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1), stokes[:,0,:], xnew)


# Interpolation back:
yq_interpol = LinearInterp1D(xnew.repeat(stokes.shape[0],1), yq_cpu)
yq_back = yq_interpol._interp(torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1),None)


# from torchinterp1d import interp1d
# out = interp1d(torch.arange(stokes.shape[-1]), stokes[:,0,:], xnew.repeat(stokes.shape[0],1))




def interpolate(xnew,x,y):
    from scipy import interpolate
    from tqdm import tqdm
    ynew = np.zeros((xnew.shape[0],xnew.shape[1]))
    for jj in tqdm(range(0,xnew.shape[0])):
        # print(x[jj,:],xnew[jj,:],y[jj,:])
        f = interpolate.interp1d(x[jj,:], y[jj,:])
        ynew[jj,:] = f(xnew[jj,:])
        # print(f(xnew[jj,:]))
        # print(ynew[jj,:])
    return ynew


def interpolates(xnew,x,y):
    from scipy import interpolate
    f = interpolate.interp1d(x, y)
    return f(xnew)

# interpolates_map = np.array(list(map(interpolates, torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1), xnew.repeat(stokes.shape[0],1),yq_cpu)))
interpolates_map = yq_back*1.01


# my_function = lambda xnew,x,y: interpolates(xnew,x,y)
# my_function(torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1), xnew.repeat(stokes.shape[0],1),yq_cpu)

# out_interpolate = interpolate(torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1), xnew.repeat(stokes.shape[0],1),yq_cpu)



# Interpolation back:
from interp_1d import CubicSpline1D
yq_interpol = CubicSpline1D(xnew.repeat(stokes.shape[0],1), yq_cpu)
yq_spline = yq_interpol._interp(torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1),None)



from weno4 import weno4
interpolates_weno4 = np.array(list(map(weno4, torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1), xnew.repeat(stokes.shape[0],1),yq_cpu)))

# interpolates_weno4 = weno4(np.arange(stokes.shape[-1]), xnew.numpy(),yq_cpu[510,:].numpy())

plt.plot(xnew,yq_cpu[510,:],'.')
# plt.plot(out[510,:],'--',label='interp1d torch')
plt.plot(stokes[510,0,:],label='original Stokes')
plt.plot(yq_back[510,:],label='interp_1d torch')
# plt.plot(yq_back__xitorch[510,:],label='yq_back__xitorch')
plt.plot(yq_spline[510,:],label='spline torch')
plt.plot(interpolates_map[510,:],'--',label='interp1d scipy')
plt.plot(interpolates_weno4[510,:],'--',label='weno4')
plt.legend()
plt.show()

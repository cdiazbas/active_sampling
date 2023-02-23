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

npoints = 3
dparam = 50.0
xnew = torch.arange(npoints)*dparam

stokes = torch.from_numpy(stokes[:,:,:].astype('float32'))
# from interp1d import interp1d
# yq_cpu = interp1d(torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1), stokes[:,0,:], xnew, None)

from interp_1d import LinearInterp1D
yq_interpol = LinearInterp1D(torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1), stokes[:,0,:])
yq_cpu = yq_interpol._interp(xnew,None)
# yq_cpu = interp1d(torch.arange(stokes.shape[-1]).repeat(stokes.shape[0],1), stokes[:,0,:], xnew, None)


plt.plot(xnew,yq_cpu[510,:],'.')
plt.plot(stokes[510,0,:])
plt.show()
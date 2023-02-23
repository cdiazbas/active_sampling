import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.convolution import convolve
from utils import line, cder
np.random.seed(seed=0)

"""
Evaluates how the inference improves depending on each method.
Coded by Carlos Diaz (UiO-RoCS, 2022)
"""


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

stokes = np.load('output/stokesI.npy')
print(stokes.shape)

stokes = np.expand_dims(stokes, axis=1)
stokes = np.concatenate((stokes,stokes[:,:,::-1])) # make that symmetric!
stokes = stokes[:,:,16:-15]
stokes = stokes[:100,:,:]
wav = np.load('wav.npy')[16:-15]
print(stokes.shape)

# Fake Stokes V under the WFA:
from tqdm import tqdm
from utils import line, cder
lin = line(8542)
C = -4.67e-13 * lin.cw**2
# Many more profiles
stokesV = np.zeros_like(stokes)
Blist = np.random.uniform(-1500,+1500,size=stokes.shape[0])
Blist = np.random.normal(0,+1500,size=stokes.shape[0])

for ii in tqdm(range(stokes.shape[0])):
    dIdw = cder(wav-lin.cw, stokes[ii,0,:][None,None,None,:])
    Blos = Blist[ii]
    stokesV[ii,:] =  C * lin.geff * Blos * dIdw


plt.figure()
plt.plot(stokes[:50,0,:].T)
plt.plot(stokesV[:50,0,:].T)
plt.savefig('output/stokes_sample.pdf')


# for jj in range(100):
for jj in [12]:

    plt.figure()
    profile = jj

    noiselevel = 1e-3
    temporallist = range(stokes.shape[-1])
    noiseI = np.random.normal(0,noiselevel,size= stokes[:,0,temporallist][:,None,None,:].shape)
    noiseV = np.random.normal(0,noiselevel,size=stokesV[:,0,temporallist].shape)
    dIdw = cder(wav[temporallist], noiseI + stokes[:,0,temporallist][:,None,None,:])
    dIdw_ = cder(wav, stokes[:,0,:][:,None,None,:])[:,:,temporallist]

    Bv = np.sum((noiseV+ stokesV[:,0,temporallist]) * dIdw[:,0,:],axis=1) / (C * lin.geff*np.sum(dIdw[:,0,:]**2.,axis=1))
    plt.plot(wav[temporallist],dIdw_[profile,0,:],label='original')
    plt.plot(wav[temporallist],dIdw[profile,0,:],label='original+noise')


    npoints = 21
    # Stokes I measured at few points: nn-guided
    temporallist = sorted(np.load('output/sampling_fullstokesI/sampling_'+str(npoints-3)+'.npy').astype('int'))
    print('nn-guided: ',temporallist)
    noiseI = np.random.normal(0,noiselevel,size= stokes[:,0,temporallist][:,None,None,:].shape)
    noiseV = np.random.normal(0,noiselevel,size=stokesV[:,0,temporallist].shape)
    dIdw = cder(wav[temporallist], noiseI + stokes[:,0,temporallist][:,None,None,:])
    Bv_nn = np.sum((noiseV+ stokesV[:,0,temporallist]) * dIdw[:,0,:],axis=1) / (C * lin.geff*np.sum(dIdw[:,0,:]**2.,axis=1))
    plt.plot(wav[temporallist],0.9*dIdw[profile,0,:],'.-',label='nn-guided')


    # Stokes I measured at few points: uniform
    temporallist = np.linspace(0.0, stokes.shape[2]-1,npoints).astype('int')
    print('uniform: ',temporallist)
    noiseI = np.random.normal(0,noiselevel,size= stokes[:,0,temporallist][:,None,None,:].shape)
    noiseV = np.random.normal(0,noiselevel,size=stokesV[:,0,temporallist].shape)
    dIdw = cder(wav[temporallist], noiseI + stokes[:,0,temporallist][:,None,None,:])
    Bv_uni = np.sum((noiseV+ stokesV[:,0,temporallist]) * dIdw[:,0,:],axis=1) / (C * lin.geff*np.sum(dIdw[:,0,:]**2.,axis=1))
    plt.plot(wav[temporallist],dIdw[profile,0,:],'.-',label='uniform')


    # Stokes I measured at few points and predicted
    stokes_predicted = np.load('output/sampling_fullstokesI/out_pred_'+str(npoints-2)+'.npy')[:stokes.shape[0],:,:]
    nnlist = np.load('output/sampling_fullstokesI/sampling_'+str(npoints-3)+'.npy').astype('int')
    temporallist = range(stokes.shape[-1])
    noiseI = np.random.normal(0,noiselevel,size= stokes[:,0,temporallist][:,None,None,:].shape)
    noiseV = np.random.normal(0,noiselevel,size=stokesV[:,0,temporallist].shape)
    dIdw = cder(wav[temporallist], noiseI + stokes_predicted[:,0,temporallist][:,None,None,:])
    print(dIdw.shape)
    Bv_pred = np.sum((noiseV[:,nnlist]+ stokesV[:,0,nnlist]) * dIdw[:,0,nnlist],axis=1) / (C * lin.geff*np.sum(dIdw[:,0,nnlist]**2.,axis=1))
    plt.plot(wav[temporallist],dIdw[profile,0,:],label='predicted')


    print('RMSE: ',np.sqrt(np.mean((Bv-Blist)**2)))
    print('RMSE nn-guided: ',np.sqrt(np.mean((Bv_nn-Blist)**2)))
    print('RMSE uniform: ',np.sqrt(np.mean((Bv_uni-Blist)**2)))
    print('RMSE predicted: ',np.sqrt(np.mean((Bv_pred-Blist)**2)))

    plt.xlabel(r'$\lambda-8542$ $[\rm \AA]$')
    plt.ylabel(r'$\rm dI/d\lambda$ [a.u.]')
    plt.minorticks_on()
    plt.legend(loc='best')
    plt.savefig('output/profiles/inference'+str(jj)+'.pdf')

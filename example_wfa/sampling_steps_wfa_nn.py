# /mn/stornext/d20/RoCS/carlosjd/projects/SAMPLING/active_sampling_testingIdeas/example5_wfa/

import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from utils import line, cder

# Sampling a spectral line:
removeedge = 15
stokes = np.load('stokesI.npy')
stokes = np.expand_dims(stokes, axis=1)[:,:,removeedge:-removeedge-1]

stokesV = np.load('stokesV.npy')
stokesV = np.expand_dims(stokesV, axis=1)[:,:,removeedge:-removeedge-1]

blos = np.load('Blist.npy')
blos = np.expand_dims(blos, axis=1)

stokes = np.concatenate((stokes,stokes[:,:,::-1])) # make that symmetric!
stokes = np.concatenate((stokes,stokes[:,:,::-1])) # make that symmetric!
stokesV = np.concatenate((stokesV,stokesV[:,:,::-1])) # make that symmetric!
stokesV = np.concatenate((stokesV,-stokesV[:,:,::-1])) # make that symmetric!
blos = np.concatenate((blos,-blos)) # make that symmetric!
blos = np.concatenate((blos,blos)) # make that symmetric!



wav = np.load('wav.npy')[removeedge:-removeedge-1]+0.0035


for i in range(15):
    plt.plot(stokes[i,0,:])
plt.savefig('stokes_sample_.pdf')

print(stokes.shape)


# STEPS:
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# x = np.array([0.0,stokes.shape[2]-1,np.argmin(np.abs(wav))])
x = np.array([np.argmin(np.abs(wav))])
npoints = 6#9
avoid = -1#3 # it is possible to avoid evaluate close points
noiselevel = 1e-3#.0#1e-2
folder = 'sampling_noise_dIdw_core_uniform'
symmetric = True



import os
if not os.path.exists(folder):
    os.makedirs(folder)
os.system('cp '+__file__.split('/')[-1]+' '+folder+'/'+__file__.split('/')[-1])


lin = line(8542)
C = -4.67e-13 * lin.cw**2


diff_point = []
for inpoint in range(len(x)+1,len(x)+npoints+1):

    if avoid < 0.5:
        availablepoints_ = np.arange(stokes.shape[-1])
    else:
        availablepoints_ = [i for i in np.arange(stokes.shape[-1]) if i not in x]

    # Removing close points to those already in the scheme
    availablepoints = [i for i in availablepoints_ if np.min(np.abs(x-i))> avoid]

    # Testing:
    qresults_chi2 = []
    for ii in tqdm(range(len(availablepoints))):
        qq = availablepoints[ii]

        if symmetric:
            # Add simmetric point:
            opposite_point = -wav[qq]
            opposite_point_index = np.argmin(np.abs(wav-opposite_point))
            qq = [qq,opposite_point_index]
            temporallist = list(np.unique(list(x)+qq).astype('int32'))
        else:  
            temporallist = list(np.unique(list(x)+[qq]).astype('int32'))
        
        # nn = 10
        # temporallist = list(np.arange(stokesV.shape[-1]))[::nn]

        try:
            noiseI = np.random.normal(0,noiselevel,size= stokes[:,0,temporallist][:,None,None,:].shape)
            noiseV = np.random.normal(0,noiselevel,size=stokesV[:,0,temporallist].shape)
            dIdw = cder(wav[temporallist], noiseI + stokes[:,0,temporallist][:,None,None,:])
            # dIdw = cder(wav, stokes[:,0,:][:,None,None,:])[:,:,temporallist]
            Bv = np.sum((noiseV+ stokesV[:,0,temporallist]) * dIdw[:,0,:],axis=1) / (C * lin.geff*np.sum(dIdw[:,0,:]**2.,axis=1))


            # fig1 = plt.figure()
            # plt.plot(temporallist, dIdw[0,0,:],'.-',zorder=-0.5,color='C0')
            # plt.plot(temporallist, dIdw2[0,0,temporallist],'.-',zorder=-0.5,color='C1')
            # plt.minorticks_on()
            # plt.savefig('sampling_nn/im_stokes_error_dIdw.pdf')
            # plt.close(fig1)
            # import sys
            # sys.exit()

            # chi2 = np.mean((Bv/1e3 - blos[:,0]/1e3)**2.,axis=0)
            chi2 = np.std((Bv/1e3 - blos[:,0]/1e3),axis=0)
            # print(np.std((Bv/1e3),axis=0))
            # print(np.std((blos[:,0]/1e3),axis=0))
            # chi2 = np.std((blos[:,0]/1e3),axis=0)
        except:
            chi2 = 1e1


        qresults_chi2.append(chi2)

    newindex = np.argmin(qresults_chi2)
    newpoint = availablepoints[newindex]


    diff_point.append([qresults_chi2,newpoint,inpoint,availablepoints,newindex])

    fig1 = plt.figure()
    plt.plot(temporallist, np.array(qresults_chi2)[temporallist],'.-',zorder=-0.5,color='k')
    plt.minorticks_on()
    plt.savefig(folder+'/im_stokes_error_temp'+str(inpoint)+'.pdf')
    plt.close(fig1)


    fig1 = plt.figure()
    for qq in range(len(diff_point)):
        plt.plot(diff_point[qq][3],diff_point[qq][0],label='it= '+str(diff_point[qq][2]),zorder=qq-0.5)
        plt.axvline(diff_point[qq][1],ls='--',color='C'+str(qq),zorder=qq)
    for jj in range(len(x)): plt.axvline(x[jj],ls='--',color='k',alpha=0.5)
    plt.legend()
    plt.yscale('log')
    plt.xlabel('Wavelength axis [index]')
    plt.ylabel('Mean inference error - LOS magnetic field [kG]')
    plt.minorticks_on()
    plt.savefig(folder+'/comb_stokes_error_temp'+str(inpoint)+'.pdf')
    plt.close(fig1)


    # fig1 = plt.figure()
    # for jj in range(len(x)): plt.axvline(x[jj],ls='--',color='k',alpha=0.5)
    # plt.plot(availablepoints, qresults_chi2,'.-',zorder=-0.5,color='k')
    # plt.axvline(newpoint,ls='--',color='k',zorder=qq)
    # plt.yscale('log')
    # plt.xlabel('Wavelength axis [index]')
    # plt.ylabel('Mean squared error - LOS magnetic field [kG]')
    # plt.minorticks_on()
    # plt.savefig('sampling_nn/im_stokes_error_temp'+str(inpoint)+'.pdf')
    # plt.close(fig1)




    print('n= '+str(len(x))+' +1 points ->',x.astype(np.int32), newpoint, '=> {0:.2e}'.format(np.max(qresults_chi2)) )#, end='\r')

    np.save(folder+'/stokes_error_temp'+str(inpoint)+'.npy',qresults_chi2[newindex])
    x = np.append(x,newpoint)

    if symmetric:
        # Add simmetric point:
        opposite_point = -wav[newpoint]
        opposite_point_index = np.argmin(np.abs(wav-opposite_point))
        x = np.append(x,opposite_point_index)
        print('After adding symmetric point:',x)
    else:
        pass


    f, ax = plt.subplots(1, 5, sharey=True,figsize=(20,5))
    for ii in range(5):
        ax[ii].plot(stokesV[ii,0,:],label='target')
        # ax[ii].plot(out[ii,0,:],label='output')
        ax[ii].scatter(x[:],stokesV[ii,0,x[:].astype('int')],color='red',label='sampling')

    plt.legend()
    plt.xlabel('Wavelength axis [index]')
    plt.ylabel('Intensity [n='+str(len(x))+' points]')
    plt.savefig(folder+'/stokes_reconstruction_temp_'+str(inpoint)+'.pdf')


# Save final sampling
np.save(folder+'/final_sampling.npy',x)

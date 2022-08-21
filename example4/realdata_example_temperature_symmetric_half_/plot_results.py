# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

ltau = np.load('ltau.npy')

fig1 = plt.figure()

c = 0
listshow = [0,5]
for ii in listshow:
    diff = np.load('sampling_nn/stokes_error_temp'+str(3+ii)+'.npy')
    diff_uniform = np.load('sampling_uniform/stokes_error_temp'+str(3+ii)+'.npy')

    plt.plot(ltau,diff,label='nn; '+str(3+2*ii)+' points',color='C'+str(c))
    plt.plot(ltau,diff_uniform,ls='--',color='C'+str(c),label='uni; '+str(3+2*ii)+' points')
    c+= 1
# plt.plot(weight.detach().numpy(),label='Weight')

plt.legend()
plt.yscale('log')
plt.xlim([-5,0.0])
plt.minorticks_on()
plt.xlabel(r'log($\tau_{500}$)')
plt.ylabel('Mean squared error - Temperature [kK]')
plt.savefig('stokes_error_temp.pdf')
plt.close(fig1)


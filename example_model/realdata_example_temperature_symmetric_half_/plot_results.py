import numpy as np
import matplotlib.pyplot as plt

"""
Plot a comparison between the different methods
Coded by Carlos Diaz (UiO-RoCS, 2022)
"""

# Includes the finalrun of the nn-guided scheme with the same type of neural network
# to be a fair comparison.

ltau = np.load('output/ltau.npy')

fig1 = plt.figure()

c = 0
listshow = [0,5]

for ii in listshow:
    diff = np.sqrt(np.load('output/sampling_nn_v2/stokes_error_temp'+str(3+ii)+'.npy'))
    diff_uniform = np.sqrt(np.load('output/sampling_uniform/stokes_error_temp'+str(3+ii)+'.npy'))

    # total points in the line
    tt = ii+3
    tt_no3dg3 = tt - 2
    tt_symm = tt_no3dg3*2
    ttfinal = tt_symm + 3

    plt.plot(ltau,diff,label='nn; '+str(ttfinal)+' points',color='C'+str(c))
    plt.plot(ltau,diff_uniform,ls='--',color='C'+str(c),label='uni; '+str(ttfinal)+' points')
    c+= 1

plt.legend()
plt.yscale('log')
plt.xlim([-5,0.0])
plt.minorticks_on()
plt.xlabel(r'log($\tau_{500}$)')
plt.ylabel('Root Mean Square Error - Temperature [kK]')
plt.savefig('output/stokes_error_temp.pdf')
plt.close(fig1)


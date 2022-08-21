# From https://raw.githubusercontent.com/burubaxair/Active-Learning/master/actreg01.py

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
np.random.seed(1)


x = np.atleast_2d(np.linspace(-10,10,101)).T

# Test parameters
mu = -0.49
sig = 3.4
amp = 0.548

def gaussian(x, mu, sig):
    return 1 - amp*np.exp(-np.square((x-mu)/sig))

# training data
x_train = np.atleast_2d(sig * np.random.randn(1,2) + mu).T
x_train = np.array([-10.,10.])[:,None]
print(x_train.shape)

def fit_GP(x_train):

    y_train = gaussian(x_train, mu, sig).ravel()

    # Instanciate a Gaussian Process model
    kernel = C(1.0, (1e-3, 1e2)) * RBF(1, (1e-2, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(x_train, y_train)

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)
    return y_train, y_pred, sigma


n_iter = 4 # number of iterations

zoom = 1.0#2
fig2, ax = plt.subplots(4,1,figsize=(4*zoom,14*zoom),sharex=True)

for i in range(n_iter):
    y_train, y_pred, sigma = fit_GP(x_train)

    ax[i].set_title('GP, Iteration '+str(i+1),fontsize=10)
    ax[i].plot(x, gaussian(x, mu, sig),label='target',color='C0')
    ax[i].scatter(x_train, y_train, s=30, marker='o',color='C3',label='sampling')
    ax[i].plot(x, y_pred,label='prediction',color='C1') # $\pm\ \sigma$
    ax[i].fill(np.concatenate([x, x[::-1]]),np.concatenate([y_pred - sigma, (y_pred  + sigma)[::-1]]), alpha=0.5, fc='C1', ec='None')
    ax[i].set_ylabel('Intensity [a.u., n='+str(2+i)+' points]')
    # ax[i].yaxis.set_ticklabels([])
    ax[i].set_ylim([0.2,1.3])
    ax[i].minorticks_on()
    ax[i].legend(loc='lower right')

    x_train = np.vstack((x_train, x[np.argmax(sigma)]))

ax[i].set_xlabel('Wavelength axis [a.u.]')
plt.savefig('im_sampling_gaussian_gp.pdf')
plt.close(fig2)
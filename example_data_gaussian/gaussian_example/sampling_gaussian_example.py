import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from tqdm import trange
np.random.seed(0)
torch.set_printoptions(sci_mode=False)

"""
Finding best sampling of a Gaussian function
Coded by Carlos Diaz (UiO-RoCS, 2022)
"""



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Synthetic dataset:
n_profiles = 20000
n_lambda = 101
noise = 0
wvl = np.linspace(-10,10,n_lambda)
v = np.random.normal(-0.0, 0.5, size=n_profiles)
dv = np.random.uniform(low=1.5, high=2.5, size=n_profiles)
amp = np.random.uniform(low=1.0, high=0.2, size=n_profiles)
stokes = 1.0 - amp[:,None]*np.exp(-(wvl[None,:]+v[:,None])**2 / dv[:,None]**2)
stokes += np.random.normal(loc=0.0, scale=noise, size=stokes.shape)

stokes = np.expand_dims(stokes, axis=1)


# Plot to check some samples in the dataset:
plt.figure()
for i in range(15):
    plt.plot(stokes[i,0,:])
plt.minorticks_on()
plt.ylabel('Intensity axis [au]')
plt.xlabel('Wavelength axis [index]')
plt.savefig('stokes_sample_.pdf')
print('stokes.shape: ',stokes.shape)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Optimization
npoints = 10
ni_epochs = 20000
diff_point = []
epsilon = 1e-8
lrnet = 5e-4
lrstep = 6

x = np.array([0.0, stokes.shape[2]-1])
newpoints2add = npoints - 2
for jj in range(newpoints2add):

    xx = x.astype('int')

    input_size = len(xx)
    output_size = stokes.shape[2]


    x_torch = torch.from_numpy(stokes[:,:,xx].astype('float32'))
    y_torch = torch.from_numpy(stokes[:,:,:].astype('float32'))

    loss_fn = nn.MSELoss()
    from resnet_model import ResidualNet
    mod = ResidualNet(input_size,output_size,hidden_features=64,num_blocks=2,database_input=x_torch,database_output=y_torch)
    optimizer = torch.optim.Adam(mod.parameters(), lr=lrnet)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(ni_epochs/lrstep), gamma=0.4)

    loss_array = []
    t = trange(ni_epochs+1, leave=True)
    for loop in t:
        optimizer.zero_grad()        #reset gradients
        out = mod(x_torch)           #evaluate model

        noise_temp = torch.from_numpy(np.random.normal(loc=0.0, scale=noise, size=stokes.shape).astype('float32'))

        loss = loss_fn(out, y_torch+noise_temp) #calculate loss
        loss.backward()              #calculate gradients
        optimizer.step()             #step fordward
        loss_array.append(loss.item())

        t.set_postfix({'loss': loss.item()})
        scheduler.step()

    
    diff = torch.mean((out - y_torch)**2.,axis=0)[0,:]
    newpoint = torch.argmax(diff).item()
    print('n= '+str(len(xx))+' (+1) points:', xx , '-> newpoint:[', newpoint, '], mse: {0:.2e}'.format(loss.item()) )#, end='\r')



    # Some plots of progress
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    diff_point.append([diff.detach().numpy()+epsilon,newpoint,jj])

    # Neural network training:
    fig1 = plt.figure()
    plt.plot(loss_array)
    plt.yscale('log')
    plt.minorticks_on()
    plt.savefig('error_'+str(jj)+'.pdf')
    plt.close(fig1)

    # Neural network prediction:
    fig1 = plt.figure()
    for qq in range(len(diff_point)):
        plt.plot(diff_point[qq][0],label='it= '+str(diff_point[qq][2]),zorder=qq-0.5)
        if qq < newpoints2add-1:
            plt.scatter(diff_point[qq][1],diff_point[qq][0][diff_point[qq][1]],zorder=qq)

    plt.legend()
    plt.yscale('log')
    plt.xlabel('Wavelength axis [index]')
    plt.ylabel('Mean squared error')
    plt.minorticks_on()
    # plt.savefig('stokes_error_'+str(jj)+'.png')
    plt.savefig('qstokes_error_'+str(jj)+'.pdf')
    plt.close(fig1)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    if jj < newpoints2add:
        x = np.append(x, newpoint)
        # print(x)  


# Saving the results
np.save('output_sampling.npy',x.astype('int'))
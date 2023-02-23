import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
torch.set_printoptions(sci_mode=False)

"""
Finding best sampling for the CaII line at 8542A
Coded by Carlos Diaz (UiO-RoCS, 2022)
"""



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Sampling a spectral line:
noise = 1e-3

# Training set
stokes = np.load('stokes.npy')
print(stokes.shape)

stokes = np.expand_dims(stokes, axis=1)
stokes = np.concatenate((stokes,stokes[:,:,::-1])) # make that symmetric!
stokes = stokes[:,:,16:-15] # make that symmetric!
stokes = stokes[:,:,:121]#[:,:,::2]

# SAMPLING DE LA CRUZ 2012
ca8_idxs = np.array([0,20,40,46,48,50,52,54,56,58,60])*2
print(len(ca8_idxs))

wav = np.load('wav.npy')[16:-15][:121]#[::2]
wav -= wav[-1] # Centrered
print('wav.shape',wav.shape)
print('wav',wav)
print(np.around(sorted(wav[ca8_idxs.astype('int')]),3))


# Plot to check some samples in the dataset:
plt.figure()
for i in range(15):
    plt.plot(wav,stokes[i,0,:])
plt.minorticks_on()
plt.ylabel('Intensity axis [au]')
plt.xlabel('Wavelength axis [index]')
plt.savefig('stokes_sample_.pdf')
print(stokes.shape)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Optimization
npoints = 11
ni_epochs = 10000#//10
diff_point = []
lrnet = 5e-4
epsilon = 1e-9
lrstep = 6




x = np.array([0.0, stokes.shape[2]-1])
newpoints2add = npoints - 2
for jj in range(newpoints2add+1):

    xx = x.astype('int')#.astype('float32')

    input_size = len(xx)
    output_size = stokes.shape[2]


    x_torch = torch.from_numpy(stokes[:,:,xx].astype('float32'))
    y_torch = torch.from_numpy(stokes[:,:,:].astype('float32'))

    loss_fn = nn.MSELoss()
    from resnet_model import ResidualNet
    mod = ResidualNet(input_size,output_size,database_input=x_torch,database_output=y_torch,hidden_features=64,num_blocks=5) #Our model 
    optimizer = torch.optim.Adam(mod.parameters(), lr=lrnet)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(ni_epochs/lrstep), gamma=0.4)


    loss_array = []
    for loop in tqdm(range(ni_epochs+1)):
        optimizer.zero_grad()        #reset gradients
        out = mod(x_torch)           #evaluate model

        noise_temp = torch.from_numpy(np.random.normal(loc=0.0, scale=noise, size=y_torch.shape).astype('float32'))

        loss = loss_fn(out, y_torch+noise_temp) #calculate loss
        loss.backward()              #calculate gradients
        optimizer.step()             #step fordward
        loss_array.append(loss.item())
        
        scheduler.step()


    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    diff = torch.mean((out - y_torch)**2.,axis=0)[0,:]

    newpoint = torch.argmax(diff).item()
    print('n= '+str(len(xx))+' (+1) points:', xx , '-> newpoint:[', newpoint, '], mse: {0:.2e}'.format(loss.item()) )


    # Some plots of progress
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    diff_point.append([diff.detach().numpy()+epsilon,newpoint,jj])

    # Neural network training:
    fig1 = plt.figure()
    plt.plot(loss_array)
    plt.title('loss_final: {0:2.2e}'.format( np.min(loss_array)))
    plt.yscale('log')
    plt.minorticks_on()
    plt.savefig('error_'+str(jj)+'.pdf')
    plt.close(fig1)


    fig1 = plt.figure()
    for qq in range(len(diff_point)):
        plt.plot(diff_point[qq][0],label='it= '+str(diff_point[qq][2]),zorder=qq-0.5)
        plt.scatter(diff_point[qq][1],diff_point[qq][0][diff_point[qq][1]],zorder=qq)

    plt.legend()
    plt.yscale('log')
    plt.xlabel('Wavelength axis [index]')
    plt.ylabel('Mean squared error [Stokes V]')
    plt.minorticks_on()
    plt.savefig('stokes_error_'+str(jj)+'.pdf')
    plt.close(fig1)


    out_test = out.detach().numpy()
    zoom = 0.8
    fig2, ax = plt.subplots(1, 5, sharey=True,figsize=(20*zoom,4.5*zoom))
    liseg = np.array([0,8,4,12,1000])
    for ii in range(len(ax)):
        ax[ii].plot(wav,100*stokes[liseg[ii],0,:],label='target')
        ax[ii].plot(wav,100*out_test[liseg[ii],0,:],label='output')
        ax[ii].scatter(wav[x.astype('int')],100*stokes[liseg[ii],0,x[:].astype('int')],color='red',label='our scheme',zorder=2,s=30.0,alpha=0.8)
        ax[ii].scatter(wav[ca8_idxs.astype('int')],100*stokes[liseg[ii],0,ca8_idxs[:].astype('int')],color='k',marker="x",label='delaCruz12',s=5.0,zorder=3)
        ax[ii].minorticks_on()
    
    ax[0].legend()
    ax[0].set_xlabel(r"$\lambda - 8542.1$ $[\rm \AA]$")
    ax[0].set_ylabel(r'Stokes V/I$\rm _C$ [%] [n='+str(2+jj)+' points]')
    plt.locator_params(axis='y', nbins=6)
    plt.ylim(-4,9.75)
    plt.savefig('stokes_reconstruction_'+str(jj)+'.pdf',bbox_inches='tight')
    plt.close(fig2)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    if jj < newpoints2add:
        x = np.append(x, newpoint)
    
    # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    print(np.around(sorted(wav[x.astype('int')]),3))
    print(np.around(sorted(wav[ca8_idxs.astype('int')]),3))
    

np.save('final_sampling.npy',x)

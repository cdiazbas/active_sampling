import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
torch.set_printoptions(sci_mode=False)

"""
Parametric model optimized with a linear interpolation method
Coded by Carlos Diaz (UiO-RoCS, 2022)
"""

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def sampling_mode(size, npoints, ninner, distance, multiple=1):
    # Generates a sampling scheme given npoints, ninner points, distance between points
    # and factor distance of outer points
    innerside = np.arange(ninner)*distance
    outterside = np.arange(npoints-ninner+1)*distance*multiple
    output = np.concatenate((size -innerside[:-1],size -(outterside+innerside[-1])), axis=0).astype('float32')
    if np.abs(np.max(output)-np.min(output)) > size:
        return None
    else:
        return  output
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Training set
stokes = np.load('output/stokes.npy')

stokes = np.expand_dims(stokes, axis=1)
stokes = np.concatenate((stokes,stokes[:,:,::-1])) # make that symmetric!
stokes = stokes[:,:,16:-15] # make that symmetric!
stokes = stokes[:,:,:121]#[:,:,::2]

wav = np.load('output/wav.npy')[16:-15][:121]#[::2]
wav -= wav[-1] # Centrered
print('wav.shape',wav.shape)
print('wav comparison:',np.around(sorted(wav[ca8_idxs.astype('int')]),3))

plt.figure()
for i in range(15):
    plt.plot(wav,stokes[i,0,:])
plt.savefig('output/stokes_sample_.pdf')
print(stokes.shape)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
npoints = 9
ni_epochs = 100000#00#//10
diff_point = []
lrnet = 5e-4
epsilon = 1e-7
lrstep = 6
noise = 1e-4
batch_size = 1000
mm = 1 # Distance factor after inner points


input_size = npoints
output_size = stokes.shape[2]
y_torch = torch.from_numpy(stokes[:,:,:].astype('float32'))
loss_fn = nn.MSELoss()
from resnet_model import ResidualNet2
mod = ResidualNet2(input_size,output_size,database_input=y_torch,database_output=y_torch,hidden_features=64,num_blocks=5,context_features=2)
optimizer = torch.optim.Adam(mod.parameters(), lr=lrnet)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(ni_epochs/lrstep), gamma=0.4)


loss_array = []
dparam_array = []
dinner_array = []
for loop in tqdm(range(ni_epochs+1)):

    # Generating random sampling points
    maxi = (output_size-1)/(npoints-1)
    sampling = None
    dparam = None
    while sampling is None:
        dparam = np.array([np.random.uniform(low=0.0, high=maxi)]).astype('float32')
        ninner = np.array([np.random.randint(low=1.0, high=npoints)]).astype('float32')
        sampling = sampling_mode(output_size-1, npoints, ninner, dparam, multiple=mm)
        dparam = torch.from_numpy(np.concatenate((dparam,ninner), axis=0))

    batch_indices = torch.LongTensor( np.random.randint(0,y_torch.shape[0],size=batch_size) )

    optimizer.zero_grad()        #reset gradients
    out = mod(y_torch[batch_indices,:,:],dparam, torch.from_numpy(sampling))           #evaluate model

    noise_temp = torch.from_numpy(np.random.normal(loc=0.0, scale=noise, size=out.shape).astype('float32'))

    loss = loss_fn(out, y_torch[batch_indices,:,:]+noise_temp) #calculate loss
    loss.backward()              #calculate gradients
    optimizer.step()             #step fordward
    loss_array.append(loss.item())
    dparam_array.append(mod.dparam[0].item())
    dinner_array.append(mod.dparam[1].item())

    scheduler.step()




# # Some plots of progress
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print('Final plots ...')
darray = np.linspace(0.0,maxi,40).astype('float32')
dinner = np.arange(1.0,npoints+1).astype('float32')
losstest_ = np.zeros((len(darray),len(dinner)))*np.nan
paramtest = []
losstest = []
for ii in range(darray.size):
    for jj in range(dinner.size):
        sampling = sampling_mode(output_size-1, npoints, dinner[jj], darray[ii], multiple=mm)
        # print(sampling)
        if sampling is not None:
            dparam = torch.from_numpy(np.concatenate((darray[ii,None],dinner[jj,None]), axis=0))
            lossitem = loss_fn(mod(y_torch,dparam, torch.from_numpy(sampling)), y_torch).item()
            losstest.append(lossitem)
            paramtest.append((darray[ii],dinner[jj]))
            losstest_[ii,jj] = lossitem
        else:
            pass

# Neural network training:
fig1 = plt.figure(figsize=(4,6))
deltawl = (wav[1]-wav[0])*1e3
plt.axhline(  deltawl*paramtest[np.argmin(losstest)][0]  ,color='black',ls='--')
plt.axvline(  paramtest[np.argmin(losstest)][1]  ,color='black',ls='--')
plt.imshow(losstest_,norm=colors.LogNorm(),cmap='magma_r',extent=[-0.5+dinner[0],dinner[-1]+0.5,deltawl*darray[0],deltawl*darray[-1]],origin='lower' , aspect='auto')
plt.title(r'Best: d={0:2.1f} $\rm m\AA$, inner={1:2.0f}, mse={2:1.1e}'.format( deltawl*paramtest[np.nanargmin(losstest)][0], paramtest[np.nanargmin(losstest)][1]  ,np.nanmin(losstest)),size=11)
plt.minorticks_on()
plt.xlabel('# inner points')
plt.ylabel(r'distance [$\rm m\AA$]')
plt.savefig('output/im_nn_'+str(mm)+'.pdf')
plt.close(fig1)


# Neural network training:
fig1 = plt.figure()
plt.plot(loss_array)
plt.axhline(  np.min(loss_array)  ,color='black',ls='--')
plt.title('loss_final: {0:2.2e}'.format( np.min(loss_array)))
plt.yscale('log')
plt.minorticks_on()
plt.savefig('output/im_error_'+str(mm)+'.pdf')
plt.close(fig1)



# Final sampling:
x = sampling_mode(output_size-1, npoints, paramtest[np.argmin(losstest)][1], paramtest[np.argmin(losstest)][0], multiple=mm)
nwav = np.interp( x, np.arange(wav.size), wav)

print('final_sampling [interal units]: ',x)
print('final_sampling [wavelength]: ',nwav)
np.save('output/final_sampling.npy',x)


dparam = torch.from_numpy(np.hstack((paramtest[np.argmin(losstest)][0], paramtest[np.argmin(losstest)][1])))
out = mod(y_torch[:,:,:], dparam, torch.from_numpy(x))  
out_test = out.detach().numpy()
zoom = 0.8
fig2, ax = plt.subplots(1, 5, sharey=True,figsize=(20*zoom,4.5*zoom))
liseg = np.array([16,18,15,21,20])
for ii in range(len(ax)):
    ax[ii].plot(wav,stokes[liseg[ii],0,:],label='target')
    ax[ii].plot(wav,out_test[liseg[ii],0,:],label='output')
    ax[ii].scatter(nwav,np.interp(nwav, wav, stokes[liseg[ii],0,:]),color='red',label='sampling DNN',zorder=2,s=30.0,alpha=0.8)
    ax[ii].minorticks_on()
    for kk in range(len(nwav)):
        ax[ii].axvline(nwav[kk],ls='-',alpha=0.1,color='k')
ax[0].legend()
ax[0].set_xlabel(r"$\lambda - 8542.1$ $[\rm \AA]$")
ax[0].set_ylabel(r'Stokes I/I$\rm _C$')
plt.locator_params(axis='y', nbins=5)
plt.savefig('output/im_stokes_'+str(mm)+'.pdf',bbox_inches='tight')
plt.close(fig2)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



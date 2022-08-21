import numpy as np
from lmfit import minimize, Parameters, report_fit
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(1)

# Create data to be fitted
methodsq = 'powell'
max_nfevq = 1000
printresult = False
n_profiles = 10000
n_lambda = 101
noise = 1e-3
wvl = np.linspace(-10,10,n_lambda)
v = np.random.normal(+0.0, 0.5, size=n_profiles)
dv = np.random.uniform(low=1.5, high=2.5, size=n_profiles)
amp = np.random.uniform(low=1.0, high=0.2, size=n_profiles)
stokes = 1.0 - amp[:,None]*np.exp(-(wvl[None,:]-v[:,None])**2 / dv[:,None]**2)
stokes += np.random.normal(loc=0.0, scale=noise, size=stokes.shape)

print('stokes.shape: ',stokes.shape)


# Sampling:
sampling_uniform = True
ploting = False

if sampling_uniform == True:
    print('Sampling: uniform')
    sampling = np.linspace(-10,10,n_lambda)
else:
    print('Sampling: network')
    sampling_nn = np.load('../gaussian_example/output_sampling.npy')


# define objective function: returns the array to be minimized
def fcn2min(params, x, data):
    model = 1.0 - params['amp']*np.exp(-(x-params['vel'])**2 / (np.abs(params['wid'])**2))
    return model - data

# create a set of Parameters
params = Parameters()
params.add('amp', value=np.mean(amp), max=1.0-0.01, min=0.01)
params.add('vel', value=np.mean(v), min=-10, max=+10)
params.add('wid', value=np.mean(dv), max=2.5, min=1.5)

n_points = 9

amp_array_gen = []
vel_array_gen = []
wid_array_gen = []

for jj in tqdm(range(2,n_points+1)):

    amp_array = []
    vel_array = []
    wid_array = []

    for ii in tqdm(range(n_profiles)):

        # Load full profile:
        data = stokes[ii,:]

        # Take only sampling values:
        if sampling_uniform == True:
            sampling = np.linspace(0,wvl.shape[0]-1,jj,dtype=np.int)
        else:
            sampling = sampling_nn[:jj]

        data = data[sampling]
        x = wvl[sampling]

        # fitting the funtion
        try:
            result = minimize(fcn2min, params, args=(x, data), method=methodsq,max_nfev=max_nfevq,calc_covar=False)
            params_fit = result.params
        except:
            # params_fit = params.copy()
            continue

        final = 1.0 - params_fit['amp']*np.exp(-(wvl-params_fit['vel'])**2 / params_fit['wid']**2)
        if printresult == True:
            report_fit(result)


        if ploting == True:
            # Plot results
            report_fit(result)
            fig1 = plt.figure()
            plt.plot(wvl, final,'--',label='output')
            plt.plot(wvl, stokes[ii,:],label='target')
            plt.scatter(x, data,color='C3',label='sampling')

            plt.legend()
            plt.xlabel('Wavelength axis [index]')
            plt.ylabel('Intensity [n='+str(jj)+' points]')
            plt.savefig('results'+str(ii)+'_'+str(jj)+'.png')
            plt.close(fig1)


        amp_array.append(params_fit['amp']-amp[ii])
        vel_array.append(params_fit['vel']-v[ii])
        wid_array.append(np.abs(params_fit['wid'])-dv[ii])

    amp_array_gen.append(np.nanstd(amp_array))
    vel_array_gen.append(np.nanstd(vel_array))
    wid_array_gen.append(np.nanstd(wid_array))

    print('sampling: ',sampling)







# Second sampling method:
sampling_uniform = False
ploting = False

if sampling_uniform == True:
    print('Sampling: uniform')
    sampling = np.linspace(-10,10,n_lambda)
else:
    print('Sampling: network')
    sampling_nn = np.load('../gaussian_example/output_sampling.npy')


amp_array_gen2 = []
vel_array_gen2 = []
wid_array_gen2 = []

for jj in tqdm(range(2,n_points+1)):

    amp_array = []
    vel_array = []
    wid_array = []

    for ii in tqdm(range(n_profiles)):

        # Load full profile:
        data = stokes[ii,:]

        # Take only sampling values:
        if sampling_uniform == True:
            sampling = np.linspace(0,wvl.shape[0]-1,jj,dtype=np.int)
        else:
            sampling = sampling_nn[:jj]

        data = data[sampling]
        x = wvl[sampling]

        # fitting the funtion
        try:
            result = minimize(fcn2min, params, args=(x, data), method=methodsq,max_nfev=max_nfevq,calc_covar=False)
            params_fit = result.params
        except:
            # params_fit = params.copy()
            continue

        final = 1.0 - params_fit['amp']*np.exp(-(wvl-params_fit['vel'])**2 / (params_fit['wid']**2))
        # report_fit(result)


        if ploting == True:
            # Plot results
            fig1 = plt.figure()
            plt.plot(wvl, final,'--',label='output')
            plt.plot(wvl, stokes[ii,:],label='target')
            plt.scatter(x, data,color='C3',label='sampling')

            plt.legend()
            plt.xlabel('Wavelength axis [index]')
            plt.ylabel('Intensity [n='+str(jj)+' points]')
            plt.savefig('results'+str(ii)+'_'+str(jj)+'.png')
            plt.close(fig1)


        amp_array.append(params_fit['amp']-amp[ii])
        vel_array.append(params_fit['vel']-v[ii])
        wid_array.append(np.abs(params_fit['wid'])-dv[ii])

    amp_array_gen2.append(np.nanstd(amp_array))
    vel_array_gen2.append(np.nanstd(vel_array))
    wid_array_gen2.append(np.nanstd(wid_array))

    print('sampling: ',sampling)



sampling_label = 'comparison' + '_'+methodsq





# Plot results
zoom = 1.2
fig2, ax = plt.subplots(3,1,figsize=(4*zoom,14/4*2.5*zoom),sharex=True)
ax[0].plot(range(2,n_points+1)[1:],amp_array_gen[1:],'.--',label='uniform scheme')
ax[0].plot(range(2,n_points+1)[1:],amp_array_gen2[1:],'.--',label='our scheme')
ax[0].legend()
# ax[0].set_xlabel('Sampling points [n]')
ax[0].set_ylabel('STD amplitude [real - fit]')
ax[0].minorticks_on()

ax[1].plot(range(2,n_points+1)[1:],vel_array_gen[1:],'.--',label='uniform scheme')
ax[1].plot(range(2,n_points+1)[1:],vel_array_gen2[1:],'.--',label='our scheme')
# ax[1].legend()
# ax[1].set_xlabel('Sampling points [n]')
ax[1].set_ylabel('STD center [real - fit]')
ax[1].minorticks_on()

ax[2].plot(range(2,n_points+1)[1:],wid_array_gen[1:],'.--',label='uniform scheme')
ax[2].plot(range(2,n_points+1)[1:],wid_array_gen2[1:],'.--',label='our scheme')
# ax[2].legend()
ax[2].set_xlabel('Sampling points [n]')
ax[2].set_ylabel('STD width [real - fit]')
ax[2].minorticks_on()

plt.tight_layout()
plt.savefig('results_comparison.pdf')
plt.close(fig2)

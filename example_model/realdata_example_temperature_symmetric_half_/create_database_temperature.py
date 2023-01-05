import numpy as np
import matplotlib.pyplot as plt
import sparsetools as sp
from tqdm import tqdm
from astropy.convolution import convolve

"""
Creating the database for the sampling process
Coded by Carlos Diaz (UiO-RoCS, 2022)
"""


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# READ THE PROFILES & STRATIFICATIONS
m = sp.model('../../../create_syntheticset/atmos_model3.nc')
s = sp.profile('../../../create_syntheticset/synthetic_out3.nc')


tempbase = m.temp.reshape(s.dat.shape[0]*s.dat.shape[1]*s.dat.shape[2],m.temp.shape[3])

database = s.dat.reshape(s.dat.shape[0]*s.dat.shape[1]*s.dat.shape[2],s.dat.shape[3],s.dat.shape[4])

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# SPECTRAL PROFILE
# We assumed a Gaussian spectral filter transmission.
def gaussian(x , s):
    return 1./np.sqrt( 2. * np.pi * s**2 ) * np.exp( -x**2 / ( 2. * s**2 ) )

ii = 0
dwav = (s.wav[1]-s.wav[0])*1000
fwhm = 100 #mA
fwhm_in_numeric_units = fwhm/dwav
print('dwav:',dwav,',fwhm:',fwhm,'[mA] ->',fwhm_in_numeric_units,'units')
sigma_in_numeric_units = fwhm_in_numeric_units/2.355 # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
xx = np.arange(-int(3*fwhm_in_numeric_units),+int(3*fwhm_in_numeric_units)+1,1.0)
print('xx array',xx)

myGaussian = gaussian(xx , sigma_in_numeric_units)
for ii in tqdm(range(database.shape[0])):
    database[ii,:,0] = convolve( database[ii,:,0], myGaussian, boundary='extend')

# Only for Stokes I
databasek = database[:,:,0]

print('->',databasek.shape)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# FILTERING SOME PROFILES

# To avoid too intense profiles that might dominate the error
databasek = databasek[np.max(databasek,axis=1) < np.max(np.mean(databasek,axis=1)+1*np.std(databasek,axis=1)),:]
databasek = databasek[np.max(databasek,axis=1) < np.max(np.mean(databasek,axis=1)+1*np.std(databasek,axis=1)),:]
print('->',databasek.shape)

nprofiles = 15000
databasek = databasek[np.random.permutation(databasek.shape[0])[:nprofiles],:]
print('->',databasek.shape)

# Taking the respective temperature for each profile:
ispectro = 0
temp_database = np.zeros((databasek.shape[0],tempbase.shape[1]))
for ispectro in tqdm(range(databasek.shape[0])):
    d = np.mean( (database[:,:,0]-databasek[ispectro,:])**2., axis=1)
    close_index = np.argmin(d)
    temp_database[ispectro,:] = tempbase[close_index,:]

np.save('temperature.npy',temp_database)
np.save('stokes.npy',databasek)


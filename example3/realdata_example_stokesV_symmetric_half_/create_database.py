import numpy as np
import matplotlib.pyplot as plt

import sparsetools as sp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.convolution import convolve
import satlas as sa


# READ THE PROFILES
s = sp.profile('../../../create_syntheticset/synthetic_out3.nc')

database = s.dat.reshape(s.dat.shape[0]*s.dat.shape[1]*s.dat.shape[2],s.dat.shape[3],s.dat.shape[4])

# We assumed a Gaussian spectral filter transmission.
def gaussian( x , s):
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
    database[ii,:,3] = convolve( database[ii,:,3], myGaussian, boundary='extend')

databasek = database[:,:,3]
print('->',databasek.shape)

# To avoid many similar samples
nprofiles = 15000//2
from sklearn.cluster import KMeans, MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=nprofiles, random_state=0, verbose=1).fit(databasek)
databasek = kmeans.cluster_centers_.copy()

databasek = np.concatenate((databasek,-databasek)) # opposite inclination

import random
databasek = databasek[random.shuffle(list(range(databasek.shape[0]))),:,:][0]
print(databasek.shape)

plt.figure()
plt.plot(databasek[:50,:].T)
plt.savefig('stokes_sample.pdf')


np.save('stokes.npy',databasek)
np.save('wav.npy',s.wav)


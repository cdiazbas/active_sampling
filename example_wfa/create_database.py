import numpy as np
import matplotlib.pyplot as plt
import sparsetools as sp
from tqdm import tqdm
from astropy.convolution import convolve

# READ THE PROFILES
s = sp.profile('../../create_syntheticset/synthetic_out3.nc')

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
    database[ii,:,0] = convolve( database[ii,:,0], myGaussian, boundary='extend')

databasek = database[:,:,0]

print('->',databasek.shape)

# To avoid too intense profiles that might dominate the error
databasek = databasek[np.max(databasek,axis=1) < np.max(np.mean(databasek,axis=1)+1*np.std(databasek,axis=1)),:]
databasek = databasek[np.max(databasek,axis=1) < np.max(np.mean(databasek,axis=1)+1*np.std(databasek,axis=1)),:]
print('->',databasek.shape)

# To avoid many similar samples
nprofiles = 15000
from sklearn.cluster import KMeans, MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=nprofiles, random_state=0, verbose=1).fit(databasek)
databasek = kmeans.cluster_centers_.copy()



# Fake Stokes V under the WFA:
from tqdm import tqdm
from utils import line, cder
lin = line(8542)
C = -4.67e-13 * lin.cw**2
# Many more profiles
databaseV = np.zeros_like(databasek)
Blist = np.random.uniform(-2000,+2000,size=databasek.shape[0])

for ii in tqdm(range(databasek.shape[0])):
    dIdw = cder(s.wav-lin.cw, databasek[ii,:][None,None,None,:])
    Blos = Blist[ii]
    databaseV[ii,:] =  C * lin.geff * Blos * dIdw


plt.figure()
plt.plot(databasek[:50,:].T)
plt.plot(databaseV[:50,:].T)
plt.savefig('stokes_sample.pdf')


np.save('stokesI.npy',databasek)
np.save('stokesV.npy',databaseV)
np.save('Blist.npy',Blist)
np.save('wav.npy',s.wav-lin.cw)


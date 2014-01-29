import string
import random
import numpy as np
import pyfits as pf

from scipy.interpolate import interp1d
from utils.focus_calcs import get_hst_focus_models
from data_manage.db_utils import get_data
from BallPeenHammer.fitting import PatchFitter

# get data from a region on the detector
xn, xx = 495, 519
yn, yx = xn, xx 
detector_size = xx - xn

label = ''.join(random.choice(string.ascii_letters + string.digits)
                for x in range(32))

try:
    dq = np.loadtxt('../data/region/dq_%d_%d_%d_%d.dat' % (xn, xx, yn, yx))
    data = np.loadtxt('../data/region/data_%d_%d_%d_%d.dat' % (xn, xx, yn, yx))
    focii = np.loadtxt('../data/region/focii_%d_%d_%d_%d.dat' %
                       (xn, xx, yn, yx))
except:
    data_dict = get_data(xn, xx, yn, yx)
    data = data_dict['pixels'] - data_dict['persist']
    dq = data_dict['dq']
    meta = data_dict['patch_meta']
    Npatches = meta.shape[0]
    mjds = np.zeros(Npatches)
    for i in range(Npatches):
        f = pf.open(meta[i, -1][1:-1])
        mjds[i] = 0.5 * (f[0].header['expstart'] + f[0].header['expend'])
        f.close()
    focii = get_hst_focus_models(mjds)
    print focii
    np.savetxt('../data/region/dq_%d_%d_%d_%d.dat' % (xn, xx, yn, yx), dq)
    np.savetxt('../data/region/data_%d_%d_%d_%d.dat' % (xn, xx, yn, yx), data)
    np.savetxt('../data/region/focii_%d_%d_%d_%d.dat' % (xn, xx, yn, yx), focii)

assert 0
patch_shape = (5, 5)
dq = dq.reshape(dq.shape[0], patch_shape[0], patch_shape[1])
data = data.reshape(dq.shape[0], patch_shape[0], patch_shape[1])

# get rid of useless patchs
minpixels = 18
ind = []
for i in range(data.shape[0]):
    test = dq[i] == 0
    if np.sum(test) >= minpixels:
        ind.append(i)

dq = dq[ind]
data = data[ind]

# initialize to tinytim
f = pf.open('../psfs/tinytim-pixelconvolved-507-507.fits') # shape (41, 41)
ini_psf = f[0].data
f.close()

# random patch centers (actual not needed if not fitting flat)
   # assign positions                                                           
xsize = (data.shape[1] - 1) / 2.
ysize = (data.shape[2] - 1) / 2.
xc = np.random.randint(detector_size - 2 * xsize, size=data.shape[0])
yc = np.random.randint(detector_size - 2 * ysize, size=data.shape[0])
xc += xsize
yc += ysize
patch_centers = (xc, yc)

# psf_grid defs
xg = np.linspace(-0.5 * patch_shape[0], 0.5 * patch_shape[0],
                  ini_psf.shape[0])
yg = xg.copy()
psf_grid = (xg, yg)

# define patch_grid
yp, xp = np.meshgrid(np.linspace(-ysize, ysize,
                                  data.shape[2]).astype(np.int),
                     np.linspace(-xsize, xsize,
                                  data.shape[1]).astype(np.int))
patch_grid = (xp, yp)

s = ['shifts', 'psf']
eps = 1.e2
ini_flat = np.ones((detector_size, detector_size))
fname = '../output/5x5psf_nll-model-trim-oldreg_%0.2f_%d_%d_%d_%d' % (np.log10(eps), 
                                                                 xn, xx, yn, yx)

import time
t = time.time()
flat, psf, shifts = PatchFitter(data, dq, ini_psf, ini_flat, patch_grid,
                                psf_grid, patch_centers, background='constant',
                                sequence=s, shift_threads=8, maxiter=5,
                                eps=eps, trim_frac=0.005,
                                ini_shifts=np.zeros((data.shape[0], 2)),
                                dumpfilebase=fname, loss_kind='nll-model',
                                floor=0.05, gain=0.01)
print time.time() - t

import os
import json
import string
import random
import numpy as np
import pyfits as pf

from scipy.interpolate import interp1d
from utils.focus_calcs import get_hst_focus_models
from data_manage.db_utils import get_data
from BallPeenHammer.fitting import PatchFitter

# parms
eps = 1.e0
gain = 0.01
floor = 0.05
s = ['shifts', 'psf']
maxiter = 5
cv_frac = 0.20
minpixels = 18
trim_frac = 0.005
loss_kind = 'nll-model'
background = 'constant'
patch_shape = (5, 5)
shift_threads = 4

# get data from a region on the detector
xn, xx = 495, 519
yn, yx = xn, xx 
detector_size = xx - xn

label = ''.join(random.choice(string.ascii_letters + string.digits)
                for x in range(16))
dqfile = '../data/region/dq_%d_%d_%d_%d.dat' % (xn, xx, yn, yx)
datafile = '../data/region/data_%d_%d_%d_%d.dat' % (xn, xx, yn, yx)
fociifile = '../data/region/focii_%d_%d_%d_%d.dat' % (xn, xx, yn, yx)
os.system('mkdir ../output/%s' % label)
fname = '../output/%s/%s' % (label, label)

try:
    dq = np.loadtxt(dqfile)
    data = np.loadtxt(datafile)
    focii = np.loadtxt(fociifile)
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

    np.savetxt(dqfile, dq)
    np.savetxt(datafile, data)
    np.savetxt(fociifile, focii)

dq = dq.reshape(dq.shape[0], patch_shape[0], patch_shape[1])
data = data.reshape(dq.shape[0], patch_shape[0], patch_shape[1])

# get rid of useless patchs
ind = []
for i in range(data.shape[0]):
    test = dq[i] == 0
    if np.sum(test) >= minpixels:
        ind.append(i)

dq = dq[ind]
data = data[ind]

# hold out fraction for CV if desired
if cv_frac > 0.0:
    Ndata = data.shape[0]
    Ntrain = np.round((1. - cv_frac) * Ndata)
    ind = np.random.random_integers(0, Ndata - 1, Ntrain)
    np.savetxt(fname + '_train_inds.dat', ind, fmt='%d')
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

# record meta data for run
runmeta = {}
runmeta['dqfile'] = dqfile
runmeta['datafile'] = datafile
runmeta['fociifile'] = fociifile
runmeta['eps'] = eps
runmeta['gain'] = gain
runmeta['floor'] = floor
runmeta['maxiter'] = maxiter
runmeta['cv_frac'] = cv_frac
runmeta['minpixels'] = minpixels
runmeta['trim_frac'] = trim_frac
runmeta['loss_kind'] = loss_kind
runmeta['background'] = background
runmeta['patch_shape'] = patch_shape
runmeta['shift_threads'] = shift_threads

# write meta data
j = json.dumps(runmeta)
f = open('../output/%s_meta.json' % label, 'w')
f.write(j)
f.close()

ini_flat = np.ones((detector_size, detector_size))

import time
t = time.time()
flat, psf, shifts = PatchFitter(data, dq, ini_psf, ini_flat, patch_grid,
                                psf_grid, patch_centers, background=background,
                                sequence=s, shift_threads=shift_threads,
                                maxiter=maxiter,
                                eps=eps, trim_frac=trim_frac,
                                ini_shifts=np.zeros((data.shape[0], 2)),
                                dumpfilebase=fname, loss_kind=loss_kind,
                                floor=floor, gain=gain)
print time.time() - t
os.system('python utils/run_list_html.py')

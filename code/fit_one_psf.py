import os
import json
import string
import random
import numpy as np
import pyfits as pf

from scipy.interpolate import interp1d
from data_manage.db_utils import get_data
from BallPeenHammer.fitting import PatchFitter
from utils.focus_calcs import get_hst_focus_models

# parms
run = 0
eps = 1.e1
gain = 0.01
floor = 0.05
s = ['shifts', 'psf']
maxiter = 0
cv_frac = 0.0
minpixels = 18
trim_frac = None
loss_kind = 'nll-model'
background = 'constant'
shift_threads = 8
patch_shape = (5, 5)
initial_psf_file = '../psfs/tinytim-pixelconvolved-507-507-5-41.fits'

clip_parms = [[1, 300], [1, 100], [1, 30], [1, 10.], [1, 6.], [1, 5.]]
#clip_parms = None

# get data from a region on the detector
xn, xx = 495, 519
yn, yx = xn, xx 
detector_size = xx - xn

# filenames, labels
label = ''.join(random.choice(string.ascii_letters + string.digits)
                for x in range(16))
dqfile = '../data/region/dq_%d_%d_%d_%d-%d.dat' % \
    (xn, xx, yn, yx, patch_shape[0])
datafile = '../data/region/data_%d_%d_%d_%d-%d.dat' % \
         (xn, xx, yn, yx, patch_shape[0])
fociifile = '../data/region/focii_%d_%d_%d_%d-%d.dat' % \
         (xn, xx, yn, yx, patch_shape[0])
fname = '../output/run%d/%s/%s' % (run, label, label)
trainindsfile = fname + '_train_inds.dat'
os.system('mkdir ../output/run%d/%s' % (run, label))

# initialize to tinytim
f = pf.open(initial_psf_file) 
ini_psf = f[0].data
f.close()

dq = np.loadtxt(dqfile)
data = np.loadtxt(datafile)
focii = np.loadtxt(fociifile)

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
    Ntest = Ndata - Ntrain
    ind = np.random.permutation(np.arange(Ndata).astype(np.int))
    traininds = ind[:Ntrain]
    np.savetxt(trainindsfile, traininds, fmt='%d')
    dq = dq[traininds]
    data = data[traininds]
    
# record meta data for run
runmeta = {}
runmeta['dqfile'] = dqfile
runmeta['datafile'] = datafile
runmeta['fociifile'] = fociifile
runmeta['trainindsfile'] = trainindsfile
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
f = open('../output/run%d/%s_meta.json' % (run, label), 'w')
f.write(j)
f.close()

ini_flat = np.ones((detector_size, detector_size))

import time
t = time.time()
flat, psf, shifts = PatchFitter(data, dq, ini_psf, ini_flat,
                                background=background,
                                sequence=s, shift_threads=shift_threads,
                                maxiter=maxiter,
                                eps=eps, trim_frac=trim_frac,
                                ini_shifts=np.zeros((data.shape[0], 2)),
                                dumpfilebase=fname, loss_kind=loss_kind,
                                floor=floor, gain=gain, clip_parms=clip_parms)
print time.time() - t
#os.system('python utils/run_list_html.py %d')

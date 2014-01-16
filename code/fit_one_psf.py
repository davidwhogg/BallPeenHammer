import numpy as np
import pyfits as pf

from data_manage.db_utils import get_data
from BallPeenHammer.fitting import PatchFitter

# get data from a region on the detector
xn, xx = 495, 519
yn, yx = xn, xx 
detector_size = xx - xn
try:
    dq = np.loadtxt('../data/dq_%d_%d_%d_%d.dat' % (xn, xx, yn, yx))
    data = np.loadtxt('../data/data_%d_%d_%d_%d.dat' % (xn, xx, yn, yx))
except:
    data_dict = get_data(xn, xx, yn, yx)
    data = data_dict['pixels'] - data_dict['persist']
    np.savetxt('../data/dq_%d_%d_%d_%d.dat' % (xn, xx, yn, yx), data_dict['dq'])
    np.savetxt('../data/data_%d_%d_%d_%d.dat' % (xn, xx, yn, yx), data)

patch_shape = (5, 5)
dq = dq.reshape(dq.shape[0], patch_shape[0], patch_shape[1])
data = data.reshape(dq.shape[0], patch_shape[0], patch_shape[1])

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

#dq *= 0.

ini_flat = np.ones((detector_size, detector_size))
PatchFitter(data, dq, ini_psf, ini_flat, patch_grid, psf_grid,
            patch_centers, background='constant', sequence=['shifts'],
            ini_shifts=np.zeros((data.shape[0], 2)), threads=8)

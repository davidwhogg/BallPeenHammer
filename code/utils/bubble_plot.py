import numpy as np
import pyfits as pf
import matplotlib.pyplot as pl

from matplotlib.colors import LogNorm
from test_fitting import *

def bubble_plot(data, shifts, filename):
    """
    Plot bubbles representing fractional flux at locations
    of shifts across the `patch`
    """
    Nbox = data.shape[1] * data.shape[2]
    Nboxx = data.shape[1]
    Nboxy = data.shape[2]
    
    f = pl.figure(figsize=(15, 15))
    pl.subplots_adjust(wspace=0., hspace=0.)

    axes = [pl.subplot(Nboxx, Nboxy, i) for i in range(Nbox)]

    for i in range(data.shape[0]):
        d = -1./np.log(data[i].ravel() / data[i].sum())
        for j in range(Nbox):
            ms = np.maximum(5, 32. * d[j])
            print ms, shifts[i]
            axes[-j].plot(shifts[i, 0], shifts[i, 1], 'o', alpha=0.5, ms=ms)

    for i in range(Nbox):
        axes[i].linewidth = 10.
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    f.savefig('../plots/' + filename)

if __name__ == '__main__':
    yn, yx = 505, 509
    xn, xx = 505, 509

    tinytimmodel = '../psfs/tinytim/tiny_k4_507_507.fits'
    f = pf.open(tinytimmodel)
    tinytimmodel = f[0].data
    f.close()

    data = get_data(xn, xx, yn, yx)
    meta = data['patch_meta']
    pixels = data['pixels']
    dq = data['dq']
    var = data['var']
    persist = data['persist']
    pixels -= persist
    var += persist ** 2.

    psf = make_psf(tinytimmodel, 0., 0., (5, 5), None)
    a, b, c = 0.897, 0.025, 0.00075
    a, b, c = 0,  0,  1.
    kernel = np.array([[a, b, a], [b, c, b], [a, b, a]])
    kernel /= kernel.sum()
    psf = convolve(psf, kernel, mode='constant')
    psf /= psf.sum()
    amps = np.max(pixels, axis=1) / psf.max()

    shifts = np.zeros((pixels.shape[0], 2))
    for i in range(pixels.shape[0]):
        print i, pixels.shape[0]
        p0 = [amps[i], 0., 0., 0., 0., 0.]
        result = fmin_bfgs(fit_star, p0, args=(pixels[i].reshape(5, 5), 
                                               dq[i].reshape(5, 5), 
                                               var[i].reshape(5, 5), 
                                               tinytimmodel))

        p = result
        shifts[i] = p[1:3]

    bubble_plot(pixels.reshape(pixels.shape[0], 5, 5), shifts, 'bubble_507.png')

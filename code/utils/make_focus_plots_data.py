import os
import psycopg2
import numpy as np
import pyfits as pf
import matplotlib.pyplot as pl

from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm
from db_utils import *

def get_mjd(filename):
    """
    Find file, open header, and return the mjd halfway 
    through the exposure.
    """
    f = pf.open(filename)
    mjd = np.array([f[0].header['EXPSTART'], f[0].header['EXPEND']]).mean()
    f.close()
    return mjd

def compute_foci(mjds):
    """
    Return the value of the focus model for given list of mjds.
    """
    # load focus model
    f = open('../data/focus-model-09-13.dat')
    l = f.readlines()
    f.close()
    mo = np.zeros(len(l))
    fo = np.zeros(len(l))
    for i, line in enumerate(l):
        line = line.split()
        mo[i] = np.float(line[0])
        fo[i] = np.float(line[-1])

    f = interp1d(mo, fo)
    ind = np.where(mjds <= mo.max())[0]
    foci = np.zeros(mjds.size) - 99.
    foci[ind] = f(mjds[ind])
    return foci

if __name__ == '__main__':
    yn, yx = 475, 525
    xn, xx = 475, 525

    data = get_data(xn, xx, yn, yx)
    meta = data['patch_meta']
    pixels = data['pixels']
    dq = data['dq']
    var = data['var']
    persist = data['persist']
    pixels -= persist
    var += persist ** 2.
    
    fluxes = np.sum(pixels, axis=1)

    m = np.median(fluxes)
    print 'median = ', m
    s = np.std(fluxes)
    print 'std = ', s

    ind = np.where(np.abs(fluxes - m) / s < 2)[0]
    fluxes = fluxes[ind]
    pixels = pixels[ind]
    meta = meta[ind]

    pixels /= fluxes[:, None]

    mjds = np.zeros(pixels.shape[0])
    for i in range(mjds.size):
        mjds[i] = get_mjd(meta[i][-1][1:-1])

    foci = compute_foci(mjds)

    ind = np.where((foci > -8.) & (foci > -8.))
    foci = foci[ind]
    pixels = pixels[ind]

    ind = np.argsort(foci)
    foci = foci[ind]
    pixels = pixels[ind]

    #i1 = foci[foci.size / 3.]
    #i2 = foci[foci.size * 2. / 3.]
    i1 = -1.
    i2 = 1.

    sk = 0.7
    f = pl.figure(figsize=(15, 10))
    pl.gray()
    pl.subplots_adjust(hspace=0.0)
    pl.subplot(231)
    ind = np.where(foci < i1)                   
    tot = np.mean(pixels[ind], axis=0)
    pl.imshow(tot.reshape(5, 5),  interpolation='nearest', origin='lower',
              norm=LogNorm(vmin=tot.min(), vmax=tot.max()))
    pl.title('focus $<\,\,%0.2f \,\mu m$' % i1)
    mn, mx = tot.min(), tot.max()
    tv = 10. ** np.linspace(np.log10(mn), np.log10(mx), 10)
    ts = ['%0.2f' % v for v in tv]
    cb = pl.colorbar(shrink=sk, ticks=tv)
    cb.ax.set_yticklabels(ts)
    pl.subplot(232)
    ind = np.where((foci >= i1) & 
                   (foci < i2))                   
    tot = np.mean(pixels[ind], axis=0)
    ref = tot.copy()
    pl.imshow(tot.reshape(5, 5),  interpolation='nearest', origin='lower',
              norm=LogNorm(vmin=tot.min(), vmax=tot.max()))
    pl.title('$%0.2f \,\mu m\,\,<$ focus $<\,\,%0.2f \,\mu m$' % (i1, i2))
    tv = 10. ** np.linspace(np.log10(mn), np.log10(mx), 10)
    ts = ['%0.2f' % v for v in tv]
    cb = pl.colorbar(shrink=sk, ticks=tv)
    cb.ax.set_yticklabels(ts)
    pl.subplot(233)
    ind = np.where(foci >= i2)                   
    tot = np.mean(pixels[ind], axis=0)
    pl.imshow(tot.reshape(5, 5),  interpolation='nearest', origin='lower',
              norm=LogNorm(vmin=tot.min(), vmax=tot.max()))
    pl.title('focus $>\,\,%0.2f \,\mu m$' % i2)
    tv = 10. ** np.linspace(np.log10(mn), np.log10(mx), 10)
    ts = ['%0.2f' % v for v in tv]
    cb = pl.colorbar(shrink=sk, ticks=tv)
    cb.ax.set_yticklabels(ts)
    pl.subplot(234)
    ind = np.where(foci < i1)                   
    tot = np.mean(pixels[ind], axis=0) / ref
    pl.imshow(tot.reshape(5, 5),  interpolation='nearest', origin='lower')
    pl.colorbar(shrink=sk)
    pl.title('top left divided by top middle')
    pl.subplot(236)
    ind = np.where(foci >= i2)                   
    tot = np.mean(pixels[ind], axis=0) / ref
    pl.imshow(tot.reshape(5, 5),  interpolation='nearest', origin='lower')
    pl.colorbar(shrink=sk)
    pl.title('top right divided by top middle')
    f.savefig('../plots/focus.png')


import numpy as np
import pyfits as pf
import matplotlib.pyplot as pl

from scipy.signal import convolve
from scipy.interpolate import RectBivariateSpline
from matplotlib.colors import LogNorm

def subsample(model, Nsamp):
    """
    Make a subsampled tiny tim model
    """
    xextent = (model.shape[0] - 1) / 2
    yextent = (model.shape[1] - 1) / 2
    xo = np.linspace(-xextent, xextent, model.shape[0])
    yo = np.linspace(-yextent, yextent, model.shape[1])
    xn = np.linspace(-xextent, xextent, Nsamp)
    yn = np.linspace(-yextent, yextent, Nsamp)

    # definitions depend on scipy version
    f = RectBivariateSpline(xo, yo, model)
    fine_model = f(yn, xn)
    return fine_model

def rebin(model, binsize):
    """
    Sum up a tinytim model in bins, aka 'pixel-convolve'
    """
    Nx = model.shape[0] / binsize
    Ny = model.shape[1] / binsize
    # yes, this is dumbass
    output = np.zeros((Nx, Ny))
    for i in range(Nx):
        for j in range(Ny):
            output[i, j] = np.sum(model[i * binsize: (i + 1) * binsize,
                                        j * binsize: (j + 1) * binsize])
    return output

if __name__ == '__main__':

    model = '../psfs/tinytim/tinytim_k4_507_507_15.fits'
    f = pf.open(model)
    model = f[0].data
    f.close()

    model = convolve(model, np.ones((5, 5)))

    Ng = 101
    Nsamp = Ng ** 2
    patch_size = 25
    halfminusone = (patch_size - 1) / 2

    # magic numbers related to tinytim output
    center = np.where(model == model.max())
    xc, yc = center[0][0], center[1][0]
    Nsubsample = 5
    delta = halfminusone * Nsubsample + 2
    model = model[xc - delta: xc + delta + 1,
                  yc - delta: yc + delta + 1]

    model = subsample(model, Nsamp)
    model = rebin(model, Ng)
    model /= model.sum()

    pl.imshow(model, interpolation='nearest', origin='lower',
              norm=LogNorm(vmin=model.min(), vmax=model.max()))
    pl.gray()
    #pl.savefig('../plots/foo.png')

    h = pf.PrimaryHDU(model)
    h.writeto('../psfs/tinytim-pixelconvolved-507-507-25-25-101-101.fits',
              clobber=True)

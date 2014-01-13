import numpy as np
import pyfits as pf
import matplotlib.pyplot as pl

from scipy.interpolate import RectBivariateSpline
from matplotlib.colors import LogNorm

def rebin(a, new_shape):
    """
    Return a rebinned array to the new shape.  `new_shape` should be an
    integer factor of the original shape.
    """
    M, N = a.shape
    m, n = new_shape
    new = a.reshape((m, M / m, n, N / n)).sum(3).sum(1)
    return new

def subsample(model, xc, yc, Nsamp, inextent=2.4, outextent=2.4):
    """
    Make a fine grid of the tinytim model
    """
    Nsteps = model.shape[0]
    xo = np.linspace(-inextent, inextent, Nsteps)
    yo = np.linspace(-inextent, inextent, Nsteps)

    xn = np.linspace(-outextent, outextent, Nsamp) - xc
    yn = np.linspace(-outextent, outextent, Nsamp) - yc

    # definitions depend on scipy version
    f = RectBivariateSpline(xo, yo, model)
    fine_model = f(yn, xn)
    return fine_model

def build_tinytim_pixel_convolved_model(model, Ng=41, psize=5.):
    """
    Take in a tinytim model, construct the pixel-convolved
    model
    """
    # magic numbers related to tinytim output
    # when subsample=5 and patch_size=5
    xc, yc = 68, 68
    Nsamp = 100
    delta = 17
    inextent = 3.4
    extent = 2.4
    model = model[xc - delta: xc + delta + 1,
                  yc - delta: yc + delta + 1]

    rng = (psize) / 2.
    xg, yg = np.meshgrid(np.linspace(-rng, rng, Ng),
                         np.linspace(rng, -rng, Ng))
    xg = xg.ravel()
    yg = yg.ravel()

    Nsteps = (Ng - 1) / np.int(psize) + 1
    step = xg[1] - xg[0]
    
    xs, ys = np.meshgrid(np.linspace((psize - 1) / 2, -(psize - 1) / 2, psize),
                         np.linspace((psize - 1) / 2, -(psize - 1) / 2, psize))
    xs = xs.ravel()
    ys = ys.ravel()

    psf_model = np.zeros(Ng ** 2)
    # this is dumbass
    for i in range(Nsteps):
        for j in range(Nsteps):

            xc = -0.5 + i * step
            yc = -0.5 + j * step
            print i, j, xc, yc

            xp = xs + xc
            yp = ys + yc

            finemodel = subsample(model, xc, yc, Nsamp, inextent=inextent)
            tinytim = rebin(finemodel, (psize, psize)).ravel()
            
            for k in range(tinytim.size):
                ind = np.where((xg == xp[k]) & (yg == yp[k]))[0]
                psf_model[ind] = tinytim[k]

    return psf_model.reshape(Ng, Ng)

if __name__ == '__main__':

    model = '../../psfs/tinytim/tiny_k4_507_507.fits'
    f = pf.open(model)
    model = f[0].data
    f.close()

    model = build_tinytim_pixel_convolved_model(model)

    h = pf.PrimaryHDU(model)
    h.writeto('../../psfs/tinytim-pixelconvolved-507-507.fits')

    f = pl.figure()
    pl.imshow(model, interpolation='nearest', origin='lower')
    f.savefig('../../plots/foo.png')

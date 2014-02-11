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

def build_tinytim_pixel_convolved_model(model, Ng=41, psize=5., Nsamp=100):
    """
    Take in a tinytim model, construct the pixel-convolved
    model
    """

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

def just_interpolate(model, extent, Ng=201):
    """
    Instead of binning, just interpolate the tinytim model
    """
    delta = (extent - 1.) / 2.
    xo = np.linspace(-delta, delta, model.shape[0])
    yo = np.linspace(-delta, delta, model.shape[1])
    xn = np.linspace(-delta, delta, Ng)
    yn = np.linspace(-delta, delta, Ng)

    # definitions depend on scipy version
    f = RectBivariateSpline(xo, yo, model)
    fine_model = f(yn, xn)
    return fine_model

    

if __name__ == '__main__':

    model = '../psfs/tinytim/tinytim_k4_507_507_15.fits'
    f = pf.open(model)
    model = f[0].data
    f.close()
    print model.shape
    # magic numbers related to tinytim output
    xc, yc = 345, 345
    delta = 12 * 5
    model = model[xc - delta: xc + delta + 1,
                  yc - delta: yc + delta + 1]


    #model = build_tinytim_pixel_convolved_model(model, Ng=101, psize=25)

    # dont do this is Ng < model.shape[0]
    model = just_interpolate(model, 25)
    model /= model.sum()

    pl.imshow(model, interpolation='nearest', origin='lower',
              norm=LogNorm(vmin=model.min(), vmax=model.max()))
    pl.gray()
    pl.savefig('../plots/foo.png')
    print model.shape

    h = pf.PrimaryHDU(model)
    h.writeto('../psfs/tinytim-pixelconvolved-507-507-25-201.fits',
              clobber=True)
    

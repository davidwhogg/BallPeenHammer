import numpy as np
import matplotlib.pyplot as pl

from Ngauss_psf import *
from scipy.special import erf
from matplotlib.colors import LogNorm
from scipy.interpolate import interp2d, RectBivariateSpline

def psf_pixel(parms, extent):
    """
    Return the pixel of a pixel-convolved PSF model using gaussians.
    
    `parms`: flattened array of amplitudes, x, y, variance_x, variance_y.
             Repeat N times for N gaussians.
    `extent`: extent of model in pixels
    """
    pval, N = 0., 0.
    rt = np.sqrt(2.)
    Ngauss = parms.size / 5
    for i in range(Ngauss):
        x0 = parms[i * 5 + 1]
        y0 = parms[i * 5 + 2]
        sx = parms[i * 5 + 3]
        sy = parms[i * 5 + 4]
        N += parms[i * 5]
        amp = 0.25 * parms[i * 5]
        pval = amp * (erf(extent[1] / rt / sx) - erf(extent[0] / rt / sx)) * \
            (erf(extent[3] / rt / sy) - erf(extent[2] / rt / sy))

    assert N == 1., 'amplitudes of gaussians dont sum to one.'
    return pval

def make_data(parms, shift, shape):
    """
    Make a patch from the PSF model.
    """
    xlower = np.linspace(-(shape[0] - 1) / 2, (shape[0] - 1) / 2, shape[0])
    ylower = np.linspace(-(shape[1] - 1) / 2, (shape[1] - 1) / 2, shape[1])
    xlower -= 0.5 - shift[0]
    ylower -= 0.5 - shift[1]
    xupper = xlower + 1.
    yupper = ylower + 1.

    xlower, ylower = np.meshgrid(xlower, ylower)
    xupper, yupper = np.meshgrid(xupper, yupper)

    xlower = xlower.ravel()
    ylower = ylower.ravel()
    xupper = xupper.ravel()
    yupper = yupper.ravel()

    patch = np.zeros_like(xlower)
    for i in range(xlower.size):
        extent = (xlower[i], xupper[i], ylower[i], yupper[i])
        patch[i] = psf_pixel(parms, extent)

    return patch.reshape(shape)

def subsample_model(model, xc, yc, Nsamp, inextent=2.4, outextent=2.4,
                    kind='cubic'):
    """
    Interpolate model onto fine grid
    """
    Nsteps = model.shape[0]
    xo = np.linspace(-inextent, inextent, Nsteps)
    yo = np.linspace(-inextent, inextent, Nsteps)

    xn = np.linspace(-outextent, outextent, Nsamp) - xc
    yn = np.linspace(-outextent, outextent, Nsamp) - yc

    # definitions depend on scipy version
    #f = interp2d(xo, yo, model.ravel().astype(np.float), kind=kind)
    f = RectBivariateSpline(xo, yo, model)
    fine_model = f(yn, xn).T
    return fine_model

def draw_fluxes(N, fluxrange, index=2):
    r = np.random.rand(N)
    na = 1. - index
    L = fluxrange[0]
    H = fluxrange[1]
    return (L ** na + r * (H ** na - L ** na)) ** (1./na)

def render_data(N, parms, fluxrange, size=500, shape=(5, 5),
                err_level=0.02):

    shifts = np.zeros((N, 2))
    fluxes = draw_fluxes(N, fluxrange)
    patches = np.zeros((N, psize, psize))

    for i in range(N):
        shifts[i] = np.random.rand(2) - 0.5
        patches[i] = make_data(parms, shifts[i], (5, 5))

    # add flux
    patches *= fluxes[:, None, None]

    # add noise
    patches += np.random.randn(N, psize, psize) * patches * err_level

    return patches, shifts, fluxes

def make_initial_model(Nsamp, psize, parms):

    model = np.zeros(Nsamp ** 2)
    
    rng = (psize) / 2.
    xs, ys = np.meshgrid(np.linspace(-rng, rng, Nsamp),
                         np.linspace(-rng, rng, Nsamp))
    xs = xs.ravel()
    ys = ys.ravel()

    for i in range(model.size):
        extent = (xs[i] - 0.5, xs[i] + 0.5,
                  ys[i] - 0.5, ys[i] + 0.5)
        model[i] = psf_pixel(parms, extent)
             
    xs = xs.reshape(Nsamp, Nsamp)
    ys = ys.reshape(Nsamp, Nsamp)
    model = model.reshape(Nsamp, Nsamp)

    return model, xs, ys

def fit_patch(model, data, flux=None):

    if flux is None:
        d = np.atleast_2d(data.ravel()).T
        m = np.atleast_2d(model.ravel()).T
        rh = np.dot(m.T, d)
        lh = np.dot(m.T, m)
        scale = rh / lh
    else:
        scale = flux

    return (d - m * scale) ** 2., scale

def all_known_except_flat(data, model, fluxes, shifts, xg, yg, 
                          detector_size, broken_row=12, flat_err = 0.05):
    """
    If all is known perfectly, how well can we recover flat?
    """
    flat = np.zeros((detector_size, detector_size))

    # assign positions
    xsize = (data.shape[1] - 1) / 2.
    ysize = (data.shape[2] - 1) / 2.
    xc = np.random.randint(detector_size - 2 * xsize, size=data.shape[0])
    yc = np.random.randint(detector_size - 2 * ysize, size=data.shape[0])
    xc += xsize
    yc += ysize

    # break the flat
    brow = xc - broken_row
    ind = np.where((brow >= -xsize) & (brow <= xsize))[0]
    for idx in ind:
        row = xsize - brow[idx]
        data[idx, row, :] *= (1. - flat_err)

    # loop over the data, update flat
    yp, xp = np.meshgrid(np.linspace(-ysize, ysize, 
                                      data.shape[2]).astype(np.int),
                         np.linspace(-xsize, xsize, 
                                      data.shape[1]).astype(np.int))
    f = RectBivariateSpline(xg, yg, ini_model)
    norm = np.zeros_like(flat)
    for i in range(data.shape[0]):
        x = np.array([-2, -1, 0, 1, 2]) + shifts[i, 0]
        y = np.array([-2, -1, 0, 1, 2]) + shifts[i, 1]
        m = f(x, y).T * fluxes[i]
        flat[xp + xc[i], yp + yc[i]] += (data[i] / m)
        norm[xp + xc[i], yp + yc[i]] += 1.
 
    return flat / norm

if __name__ == '__main__':
    
    np.random.seed(100)

    detector_size = 25
    N = 25 * detector_size ** 2
    Nsamp = 41
    psize = 5
    fluxrange = (8., 64.0)
    
    wfc_ir_psf = 1.176 / 2. / np.sqrt(2. * np.log(2.))
    parms = np.array([1, 0., 0., wfc_ir_psf, wfc_ir_psf])
    ini_model, xg, yg = make_initial_model(Nsamp, psize, parms)

    data, shifts, fluxes = render_data(N, parms, fluxrange, err_level=0.02)

    flat = all_known_except_flat(data, ini_model, fluxes, shifts,
                                 xg[0], yg[:, 0], detector_size)

    f = pl.figure()
    pl.imshow(flat, interpolation='nearest', origin='lower')
    pl.colorbar()
    f.savefig('../plots/foo.png')

    assert 0
    f = RectBivariateSpline(xg[0], yg[:, 0], ini_model)
    x = np.array([-2, -1, 0, 1, 2]) + shifts[0, 0]
    y = np.array([-2, -1, 0, 1, 2]) + shifts[0, 1]
    m = f(x, y).T * fluxes[0]

    f = pl.figure(figsize=(15, 5))
    pl.subplot(131)
    pl.imshow(data[0], interpolation='nearest', origin='lower',
              norm=LogNorm(vmin=data[0].min(), vmax=data[0].max()))
    pl.colorbar()
    pl.subplot(132)
    pl.imshow(m, interpolation='nearest', origin='lower',
              norm=LogNorm(vmin=data[0].min(), vmax=data[0].max()))
    pl.colorbar()
    pl.subplot(133)
    pl.imshow((data[0]-m), interpolation='nearest', origin='lower')
    pl.colorbar()
    f.savefig('../plots/foo.png')

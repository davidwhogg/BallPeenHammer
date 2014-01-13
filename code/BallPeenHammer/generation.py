import numpy as np

from scipy.special import erf
from scipy.interpolate import RectBivariateSpline

def get_gaussian_psf_pixel(parms, extent):
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

def render_psfs(psf_model, shifts, data_shape, xpsf, ypsf):
    """
    Make psfs under current model.
    """
    interp_func = RectBivariateSpline(xpsf, ypsf, psf_model)
    psfs = np.zeros(data_shape)

    # patch shape is odd, should be checked elsewhere...
    dx = data_shape[0] - 1
    dy = data_shape[0] - 1
    x = np.arange(-dx, dx, data_shape[0]).astype(np.int)
    y = np.arange(-dy, dy, data_shape[1]).astype(np.int)

    for i in range(psfs.shape[0]):
        psfs[i] = interp_func(x + shifts[i, 0], y + shifts[i, 1])

    return psfs

def make_patch(parms, shift, shape):
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
        patch[i] = get_gaussian_psf_pixel(parms, extent)

    return patch.reshape(shape)

def draw_fluxes(N, fluxrange, index=2):
    """
    Draw fluxes from a power law.
    """
    r = np.random.rand(N)
    na = 1. - index
    L = fluxrange[0]
    H = fluxrange[1]
    return (L ** na + r * (H ** na - L ** na)) ** (1./na)

def render_data(N, parms, fluxrange, size=500, shape=(5, 5),
                err_level=0.02):
    """
    Given number of patchs, gaussian psf parameters, and flux range 
    render fake data.
    """
    shifts = np.zeros((N, 2))
    fluxes = draw_fluxes(N, fluxrange)
    patches = np.zeros((N, shape[0], shape[1]))

    for i in range(N):
        shifts[i] = np.random.rand(2) - 0.5
        patches[i] = make_patch(parms, shifts[i], (shape[0], shape[1]))

    # add flux
    patches *= fluxes[:, None, None]

    # add noise
    patches += np.random.randn(N, shape[0], shape[1]) * patches * err_level

    return patches, shifts, fluxes

def make_initial_gaussian_psf(Nsamp, psize, parms):
    """
    Make a pixel-convolved psf model sampled by Nsamp 
    in each direction, given patch size and gaussian 
    parameters.
    """
    model = np.zeros(Nsamp ** 2)
    
    rng = (psize) / 2.
    xs, ys = np.meshgrid(np.linspace(-rng, rng, Nsamp),
                         np.linspace(-rng, rng, Nsamp))
    xs = xs.ravel()
    ys = ys.ravel()

    for i in range(model.size):
        extent = (xs[i] - 0.5, xs[i] + 0.5,
                  ys[i] - 0.5, ys[i] + 0.5)
        model[i] = get_gaussian_psf_pixel(parms, extent)
             
    xs = xs.reshape(Nsamp, Nsamp)
    ys = ys.reshape(Nsamp, Nsamp)
    model = model.reshape(Nsamp, Nsamp)
    
    return model, xs, ys

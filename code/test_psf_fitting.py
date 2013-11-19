import numpy as np
import matplotlib.pyplot as pl

from matplotlib.colors import LogNorm
from scipy.interpolate import interp2d, RectBivariateSpline

def psf_pixel(parms, extent, size=200):
    """
    Return the pixel of a pixel-convolved PSF model using gaussians.
    
    `parms`: flattened array of amplitudes, x, y, variance_x, variance_y.
             Repeat N times for N gaussians.
    `extent`: extent of model in pixels
    `size`: subsampled size
    """
    xg, yg = np.meshgrid(np.linspace(extent[0], extent[1], size),
                         np.linspace(extent[2], extent[3], size))
    Ngauss = parms.size / 5
    psf = np.zeros((size, size))
    for i in range(Ngauss):
        x0 = parms[i * 5 + 1]
        y0 = parms[i * 5 + 2]
        vx = parms[i * 5 + 3]
        vy = parms[i * 5 + 4]
        amp = parms[i * 5] / (2. * np.pi * np.sqrt(vx) * np.sqrt(vy))
        psf += amp * np.exp(-0.5 * ((xg - x0) ** 2 / vx + (yg - y0) ** 2 / vy))
    return psf.sum() / size ** 2.


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
    t = time.time()
    fine_model = f(yn, xn).T
    return fine_model

def draw_fluxes(N, fluxrange, index=2):
    r = np.random.rand(N)
    na = 1. - index
    L = fluxrange[0]
    H = fluxrange[1]
    return (L ** na + r * (H ** na - L ** na)) ** (1./na)

def render_data(N, parms, fluxrange, size=500, extent=(-2.5, 2.5),
                err_level=0.02):

    psize = np.int(extent[1] - extent[0])
    shifts = np.zeros((N, 2))
    fluxes = draw_fluxes(N, fluxrange)
    patches = np.zeros((N, psize, psize))

    for i in range(N):
        input_parms = parms.copy()
        shifts[i] = np.random.rand(2) - 0.5
        for j in range(input_parms.size):
            if np.mod(j, 5) == 1:
                input_parms[j:j+2] += shifts[i]
        
        subsampled = make_subsampled(input_parms, extent, size)
        patches[i] = rebin(subsampled, (psize, psize))

    # normalize and add flux
    patches /= patches.sum(axis=1).sum(axis=1)[:, None, None]
    patches *= fluxes[:, None, None]

    # noise
    patches += np.random.randn(35, 5, 5) * patches * err_level

    return patches, shifts, fluxes

def make_initial_model(Nsamp, psize, size=500):

    model = np.zeros(Nsamp ** 2)
    wfc_ir_psf = 1.176 / 2. / np.sqrt(2. * np.log(2.))
    
    rng = (psize) / 2.
    xs, ys = np.meshgrid(np.linspace(-rng, rng, Nsamp),
                         np.linspace(-rng, rng, Nsamp))
    xs = xs.ravel()
    ys = ys.ravel()

    parms = np.array([1, 0., 0.0, wfc_ir_psf ** 2., 
                      wfc_ir_psf ** 2.])
    for i in range(model.size):
        extent = (xs[i] - 0.5, xs[i] + 0.5,
                  ys[i] - 0.5, ys[i] + 0.5)
        model[i] = psf_pixel(parms, extent)
             
    return model.reshape(Nsamp, Nsamp), xs, ys

if __name__ == '__main__':
    
    N = 35 * 1
    Nsamp = 41
    psize = 5
    fluxrange = (8., 64.0)

    wfc_ir_psf = 1.176 / 2. / np.sqrt(2. * np.log(2.))
    parms = np.array([1, 0., 0., wfc_ir_psf ** 2., wfc_ir_psf ** 2.])

    ini_model, xs, ys = make_initial_model(Nsamp, psize)
    print ini_model.sum()
    f = pl.figure()
    pl.gray()
    pl.imshow(ini_model, interpolation='nearest', origin='lower',
              extent=(-2.5, 2.5, -2.5, 2.5), 
              norm=LogNorm(vmin=1.e-6, vmax=ini_model.max()))
    pl.colorbar()
    f.savefig('../plots/foo.png')
    assert 0
    data, shifts, fluxes = render_data(N, parms, fluxrange)

    f = pl.figure()
    pl.gray()
    pl.imshow(data[0], interpolation='nearest')
    pl.title('%0.2f %0.2f' % (shifts[0, 0], shifts[0, 1]))
    f.savefig('../plots/foo.png')

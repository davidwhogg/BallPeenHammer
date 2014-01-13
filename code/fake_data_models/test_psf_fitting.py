import numpy as np
import matplotlib.pyplot as pl

from matplotlib.colors import LogNorm
from scipy.optimize import fmin_bfgs
from scipy.interpolate import RectBivariateSpline

from generation import *
from fitting import *

def run_test(data, model, fluxes, shifts, xg, yg, detector_size, 
             broken_row=12, flat_err = 0.05, kind='all'):
    """
    Return flat after fitting stars.
    """
    ini_flat = np.ones((detector_size, detector_size))

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
    interp_func = RectBivariateSpline(xg, yg, ini_model)

    if kind == 'all':
        flat = np.zeros((detector_size, detector_size))
        norm = np.zeros_like(flat)
        for i in range(data.shape[0]):
            model = fit_known_shift_and_flux(interp_func, shifts[i], fluxes[i])
            flat[xp + xc[i], yp + yc[i]] += (data[i] / model)
            norm[xp + xc[i], yp + yc[i]] += 1.
        flat /= norm

    if kind == 'psf':
        flat, norm, tse = fit_known_psf(data,
                                        ini_flat, xc, xp, yc, yp,
                                        interp_func=interp_func)
        return flat

    if kind == 'psf-shifts':
        flat, norm, tse = fit_known_psf_and_shift(data, shifts, 
                                        ini_flat, xc, xp, yc, yp,
                                                  interp_func=interp_func)
        return flat

    if kind == 'shifts':
        running_flat = ini_flat.copy()
        result = fmin_bfgs(sqe_known_shift, ini_model.ravel(),
                           args=(data, shifts, ini_flat, xc,
                                 xp, yc, yp, xg, yg, running_flat), maxiter=1)

        return running_flat, result


if __name__ == '__main__':
    
    np.random.seed(1040)

    detector_size = 15
    N = 40 * detector_size ** 2
    Nsamp = 41
    psize = 5
    fluxrange = (8., 64.0)
    
    wfc_ir_psf = 1.176 / 2. / np.sqrt(2. * np.log(2.))
    parms = np.array([1, 0., 0., wfc_ir_psf, wfc_ir_psf])
    ini_model, xg, yg = make_initial_model(Nsamp, psize, parms)

    data, shifts, fluxes = render_data(N, parms, fluxrange, err_level=0.0)

    shifts += 0.005 * np.random.randn(shifts.shape[0], 2)

    flat, psf = run_test(data, ini_model, fluxes, shifts,
                         xg[0], yg[:, 0], detector_size, kind='shifts', 
                         broken_row=8, flat_err=0.05)

    #flat *= flat.size / flat.sum()
    f = pl.figure()
    pl.imshow(flat, interpolation='nearest', origin='lower')
    pl.colorbar()
    f.savefig('../../plots/vary_psf_one_inter-flat.png')
    psf = psf.reshape(Nsamp, Nsamp) 
    f = pl.figure()
    pl.imshow(psf, interpolation='nearest', origin='lower')
    pl.colorbar()
    f.savefig('../../plots/vary_psf_one_inter-psf.png')


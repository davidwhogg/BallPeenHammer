import numpy as np
import matplotlib.pyplot as pl

from matplotlib.colors import LogNorm
from scipy.optimize import fmin_bfgs
from scipy.interpolate import RectBivariateSpline

from generation import *
from fitting import *

def run_test(data, model, fluxes, shifts, xg, yg, detector_size, 
             broken_row=12, flat_err = 0.05):
    """
    Return flat after fitting stars.
    """
    ini_flat = np.ones((detector_size, detector_size))
    true_flat = np.ones((detector_size, detector_size))

    # assign positions
    xsize = (data.shape[1] - 1) / 2.
    ysize = (data.shape[2] - 1) / 2.
    xc = np.random.randint(detector_size - 2 * xsize, size=data.shape[0])
    yc = np.random.randint(detector_size - 2 * ysize, size=data.shape[0])
    xc += xsize
    yc += ysize

    # break the flat
    true_flat[broken_row, :] *= (1. - flat_err)
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

    flat, norm, tse = fit_known_psf_and_shift(data, shifts, 
                                              ini_flat, xc, xp, yc, yp,
                                              interp_func=interp_func)
    return flat, true_flat

def two_panel_plot_flats(flat, true_flat, base):
    """
    Plot the flats
    """
    f = pl.figure()
    pl.subplot(121)
    pl.imshow(flat, interpolation='nearest', origin='lower')
    pl.colorbar(shrink=0.5)
    pl.title('inferred flat')
    pl.subplot(122)
    pl.imshow(flat / true_flat, interpolation='nearest', origin='lower')
    pl.colorbar(shrink=0.5)
    pl.title('inferred flat / true flat')
    f.savefig('../../plots/%s.png' % base)

def two_panel_plot_meanstd(meanflat, stdflat, base):
    """
    Plot the flats
    """
    
    f = pl.figure()
    pl.subplot(121)
    print meanflat.shape
    pl.imshow(meanflat, interpolation='nearest', origin='lower')
    pl.colorbar(shrink=0.5)
    pl.title('mean flat')
    pl.subplot(122)
    pl.imshow(stdflat, interpolation='nearest', origin='lower')
    pl.colorbar(shrink=0.5)
    pl.title('standard deviation flat')
    f.savefig('../../plots/%s.png' % base)


if __name__ == '__main__':
    

    Nreal = 10
    detector_size = 15
    psize = 5
    fluxrange = (8., 64.0)
    
    wfc_ir_psf = 1.176 / 2. / np.sqrt(2. * np.log(2.))
    parms = np.array([1, 0., 0., wfc_ir_psf, wfc_ir_psf])

    Ns = [60, 50, 40, 30, 20, 10]
    Nsamps = [61, 51, 41, 31, 21]
    errs = [0., 1e-4, 5e-3, 1e-3, 5e-3, 1e-2, 5e-2]
    shifterrs = [0., 1e-4, 5e-3, 1e-3, 5e-3, 1e-2, 5e-2]

    kind = 'noise'
    for i in range(len(errs)):
        N = Ns[2] * detector_size ** 2
        Nsamp = Nsamps[0]
        ini_model, xg, yg = make_initial_model(Nsamp, psize, parms)
        flats = np.zeros((Nreal, detector_size, detector_size))
        for j in range(Nreal):
            np.random.seed(j)
            data, shifts, fluxes = render_data(N, parms, fluxrange,
                                               err_level=errs[i])

            flat, true_flat = run_test(data, ini_model, fluxes, shifts,
                                       xg[0], yg[:, 0], detector_size, 
                                       broken_row=8, flat_err=0.05)

            flats[i] = flat
            base = kind + '%f-%d' % (errs[i], j)
            two_panel_plot_flats(flat, true_flat, base)
            
        base = kind + '%f-meanstd' % errs[i]
        two_panel_plot_meanstd(flats.mean(axis=0), flats.std(axis=0), base)

    kind = 'Ndata'
    for i in range(len(Ns)):
        N = Ns[i] * detector_size ** 2
        Nsamp = Nsamps[0]
        ini_model, xg, yg = make_initial_model(Nsamp, psize, parms)
        flats = np.zeros((Nreal, detector_size, detector_size))
        for j in range(Nreal):
            np.random.seed(j)
            data, shifts, fluxes = render_data(N, parms, fluxrange,
                                               err_level=errs[0])

            flat, true_flat = run_test(data, ini_model, fluxes, shifts,
                                       xg[0], yg[:, 0], detector_size, 
                                       broken_row=8, flat_err=0.05)

            flats[i] = flat
            base = kind + '%f-%d' % (N, j)
            two_panel_plot_flats(flat, true_flat, base)
            
        base = kind + '%f-meanstd' % N
        two_panel_plot_meanstd(flats.mean(axis=0), flats.std(axis=0), base)

    kind = 'shift'
    for i in range(len(shifterrs)):
        N = Ns[2] * detector_size ** 2
        Nsamp = Nsamps[0]
        ini_model, xg, yg = make_initial_model(Nsamp, psize, parms)
        flats = np.zeros((Nreal, detector_size, detector_size))
        for j in range(Nreal):
            np.random.seed(j)
            data, shifts, fluxes = render_data(N, parms, fluxrange,
                                               err_level=errs[0])
            s = shifts + shifterrs[i] * np.random.randn(shifts.shape[0], 2)

            flat, true_flat = run_test(data, ini_model, fluxes, s,
                                       xg[0], yg[:, 0], detector_size, 
                                       broken_row=8, flat_err=0.05)

            flats[i] = flat
            base = kind + '%f-%d' % (shifterrs[i], j)
            two_panel_plot_flats(flat, true_flat, base)
            
        base = kind + '%f-meanstd' % shifterrs[i]
        two_panel_plot_meanstd(flats.mean(axis=0), flats.std(axis=0), base)

    kind = 'Nsamp'
    for i in range(len(Nsamps)):
        N = Ns[2] * detector_size ** 2
        Nsamp = Nsamps[i]
        ini_model, xg, yg = make_initial_model(Nsamp, psize, parms)
        flats = np.zeros((Nreal, detector_size, detector_size))
        for j in range(Nreal):
            np.random.seed(j)
            data, shifts, fluxes = render_data(N, parms, fluxrange,
                                               err_level=errs[0])

            flat, true_flat = run_test(data, ini_model, fluxes, shifts,
                                       xg[0], yg[:, 0], detector_size, 
                                       broken_row=8, flat_err=0.05)

            flats[i] = flat
            base = kind + '%f-%d' % (Nsamp, j)
            two_panel_plot_flats(flat, true_flat, base)
            
        base = kind + '%f-meanstd' % Nsamp
        two_panel_plot_meanstd(flats.mean(axis=0), flats.std(axis=0), base)

    kind = 'psf'
    for i in range(len(errs)):
        N = Ns[2] * detector_size ** 2
        Nsamp = Nsamps[0]
        ini_model, xg, yg = make_initial_model(Nsamp, psize, parms)
        flats = np.zeros((Nreal, detector_size, detector_size))
        for j in range(Nreal):
            np.random.seed(j)
            data, shifts, fluxes = render_data(N, parms, fluxrange,
                                               err_level=errs[0])

            im = ini_model + errs[i] * np.random.randn(ini_model.shape[0],
                                                       ini_model.shape[1]) 
            flat, true_flat = run_test(data, im, fluxes, shifts,
                                       xg[0], yg[:, 0], detector_size, 
                                       broken_row=8, flat_err=0.05)

            flats[i] = flat
            base = kind + '%f-%d' % (errs[i], j)
            two_panel_plot_flats(flat, true_flat, base)
            
        base = kind + '%f-meanstd' % errs[i]
        two_panel_plot_meanstd(flats.mean(axis=0), flats.std(axis=0), base)
        

import multiprocessing
import numpy as np

from .grid_definitions import get_grids
from .generation import render_psfs
from scipy.optimize import fmin_powell
from .patch_fitting import fit_single_patch, data_loss

def update_shifts(data, dq, current_flat, current_psf, patch_shape,
                  patch_centers, ref_shifts, background, threads, loss_kind,
                  floor, gain, clip_parms):
    """
    Update the estimate of the subpixel shifts, given current psf and flat.
    """
    psf_grid, patch_grid = get_grids(current_flat.shape, patch_shape,
                                     current_psf.shape,
                                     core_shape=data[0].shape)
    if patch_centers is None:
        c = np.ones(data.shape[0]).astype(np.int) * (patch_shape[0] + 1)/ 2
        patch_centers = (c, c)

    xpg, ypg = patch_grid[0], patch_grid[1]
    xpc, ypc = patch_centers[0], patch_centers[1]

    p0 = [0., 0.]
    ssqe = np.zeros(data.shape[0])
    shifts = np.zeros((data.shape[0], 2))
    if threads == 1:
        for i in range(data.shape[0]):
            flat = current_flat[xpg + xpc[i], ypg + ypc[i]]
            res = update_single_shift((p0, current_psf, data[i], dq[i], 
                                       flat, psf_grid, ref_shifts[i],
                                       background, loss_kind, floor, gain,
                                       clip_parms))

            shifts[i] = res[0] + ref_shifts[i]
            ssqe[i] = res[1]
    else:
        pool = multiprocessing.Pool(threads)
        mapfn = pool.map
        argslist = [None] * data.shape[0]
        for i in range(data.shape[0]):
            flat = current_flat[xpg + xpc[i], ypg + ypc[i]]
            argslist[i] = [p0, current_psf, data[i], dq[i], flat,
                           psf_grid, ref_shifts[i], background, loss_kind,
                           floor, gain, clip_parms]

        results = list(mapfn(update_single_shift, [args for args in argslist]))
        for i in range(data.shape[0]):
            shifts[i] = results[i][0] + ref_shifts[i]
            ssqe[i] = results[i][1]

        pool.close()
        pool.terminate()
        pool.join()

    return shifts, ssqe / data.shape[0]

def update_single_shift((p0, current_psf, datum, dq, flat, psf_grid, ref_shift,
                         background, loss_kind, floor, gain, clip_parms)):
    """
    Update a single shift
    """
    # fmin or fmin_powell seems to perform better than
    # fmin_bfgs or fmin_l_bfgs_b.  Powell seems to be as
    # good as fmin, and quicker.
    res = fmin_powell(shift_loss, p0, full_output=True, disp=False,
               args=(current_psf, datum, dq, flat, psf_grid, ref_shift,
                     background, loss_kind, floor, gain, clip_parms))
    return res

def shift_loss(delta_shift, psf_model, data, dq, flat, psf_grid, ref_shift, 
               background, loss_kind, floor, gain, clip_parms):
    """
    Evaluate the shift for a given patch.
    """
    shift = delta_shift + ref_shift

    # Horrible hack for minimizers w/o bounds
    if (np.abs(shift[0]) > 0.5) | (np.abs(shift[1]) > 0.5):
        return 1.e10

    shape = (1, data.shape[0], data.shape[1])
    psf = render_psfs(psf_model, np.array([shift]), shape, psf_grid[0],
                      psf_grid[1])

    psf = psf.ravel()
    data = data.ravel()
    flat = flat.ravel()

    flux, bkg_parms, bkg, ind = fit_single_patch((data, psf, flat, dq.ravel(),
                                                  background, floor, gain, 
                                                  clip_parms))

    model = flat * (flux * psf + bkg)
    ssqe = data_loss(data[ind], model[ind], loss_kind, floor, gain)

    return np.sum(ssqe)

def diagnostic_plot(data, model, floor, gain, patch_shape=(5, 5)):
    """
    Quick and dirty plot to check things are ok
    """
    import matplotlib.pyplot as pl
    f = pl.figure(figsize=(12, 4))
    pl.subplot(131)
    pl.imshow(data.reshape(patch_shape), interpolation='nearest',
              origin='lower')
    pl.colorbar()
    pl.subplot(132)
    pl.imshow(model.reshape(patch_shape), interpolation='nearest',
              origin='lower')
    pl.colorbar()
    pl.subplot(133)
    var = floor + gain * np.abs(model)
    pl.imshow(((data - model) ** 2. / var).reshape(patch_shape),
              interpolation='nearest', origin='lower')
    pl.colorbar()
    f.savefig('../plots/foo.png')
    assert 0

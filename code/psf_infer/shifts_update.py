import multiprocessing
import numpy as np

from .grid_definitions import get_grids
from .generation import render_psfs
from scipy.optimize import fmin_powell
from .patch_fitting import evaluate

def update_shifts(data, dq, psf_model, patch_shape, ref_shifts, background,
                  Nthreads, loss_kind, floor, gain, clip_parms):
    """
    Update the estimate of the subpixel shifts, given current psf and flat.
    """
    # grid defs
    psf_grid, patch_grid = get_grids(patch_shape, psf_model.shape,
                                     core_shape=patch_shape)
    # initialize
    p0 = [0., 0.]
    ssqe = np.zeros(data.shape[0])
    shifts = np.zeros((data.shape[0], 2))

    # map to threads
    pool = multiprocessing.Pool(Nthreads)
    mapfn = pool.map
    argslist = [None] * data.shape[0]
    for i in range(data.shape[0]):
        argslist[i] = [p0, psf_model, data[None, i], dq[None, i], psf_grid,
                       ref_shifts[None, i], patch_shape, background, loss_kind,
                       floor, gain, clip_parms]

    results = list(mapfn(update_single_shift, [args for args in argslist]))
    for i in range(data.shape[0]):
        shifts[i] = results[i][0] + ref_shifts[i]
        ssqe[i] = results[i][1]

    pool.close()
    pool.terminate()
    pool.join()

    return shifts, ssqe / data.shape[0]

def update_single_shift((p0, psf_model, datum, dq, psf_grid, ref_shift,
                         patch_shape, background, loss_kind, floor, gain,
                         clip_parms)):
    """
    Update a single shift
    """
    # fmin or fmin_powell seems to perform better than
    # fmin_bfgs or fmin_l_bfgs_b.  Powell seems to be as
    # good as fmin, and quicker.
    res = fmin_powell(shift_loss, p0, full_output=True, disp=False,
               args=(psf_model, datum, dq, psf_grid, ref_shift, patch_shape,
                     background, loss_kind, floor, gain, clip_parms))
    return res

def shift_loss(delta_shift, psf_model, datum, dq, psf_grid, ref_shift, 
               patch_shape, background, loss_kind, floor, gain, clip_parms):
    """
    Evaluate the shift for a given patch.
    """
    shift = delta_shift + ref_shift

    # Horrible hack for minimizers w/o bounds
    if np.any(np.abs(shift) > 0.5):
        return 1.e10

    ssqe = evaluate((datum, dq, shift, psf_model, psf_grid, patch_shape,
                     background, floor, gain, clip_parms, loss_kind))

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

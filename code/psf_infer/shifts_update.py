import multiprocessing
import numpy as np

from .grid_definitions import get_grids
from .generation import render_psfs
from scipy.optimize import fmin_powell
from .patch_fitting import evaluate

def update_shifts(data, dq, psf_model, ref_shifts, parms):
    """
    Update the estimate of the subpixel shifts, given current psf and flat.
    """
    Ndata = parms.Ndata

    # initialize
    p0 = [0., 0.]
    ssqe = np.zeros(Ndata)
    shifts = np.zeros((Ndata, 2))

    # map to threads
    pool = multiprocessing.Pool(parms.Nthreads)
    mapfn = pool.map
    argslist = [None] * Ndata
    for i in range(Ndata):
        argslist[i] = [p0, psf_model, data[None, i], dq[None, i],
                       ref_shifts[None, i], parms]

    results = list(mapfn(update_single_shift, [args for args in argslist]))
    for i in range(Ndata):
        shifts[i] = results[i][0] + ref_shifts[i]
        ssqe[i] = results[i][1]

    pool.close()
    pool.terminate()
    pool.join()

    return shifts, ssqe / Ndata

def update_single_shift((p0, psf_model, datum, dq, ref_shift, parms)):
    """
    Update a single shift
    """
    # fmin or fmin_powell seems to perform better than
    # fmin_bfgs or fmin_l_bfgs_b.  Powell seems to be as
    # good as fmin, and quicker.
    res = fmin_powell(shift_loss, p0, full_output=True, disp=False,
               args=(psf_model, datum, dq, ref_shift, parms))
    return res

def shift_loss(delta_shift, psf_model, datum, dq, ref_shift, parms):
    """
    Evaluate the shift for a given patch.
    """
    shift = delta_shift + ref_shift

    # Horrible hack for minimizers w/o bounds
    if np.any(np.abs(shift) > 0.5):
        return 1.e10

    ssqe = evaluate((datum, dq, shift, psf_model, parms, True))

    return np.sum(ssqe)

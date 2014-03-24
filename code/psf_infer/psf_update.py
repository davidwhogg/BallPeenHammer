import multiprocessing
import numpy as np

from .grid_definitions import get_grids
from .generation import render_psfs
from .patch_fitting import fit_single_patch, data_loss
from scipy.optimize import fmin_powell


def update_psf(data, dq, psf_model, patch_shape, shifts, background, eps,
               loss_kind, floor, gain, clip_parms, Nthreads, h,
               old_psf_steps=None, old_total_cost=np.inf, small=1.e-16, 
               tol=1.e-1, rate=2., ini_step=0.1):
    """
    Update the psf model, using bfgs.
    """
    global count
    count = 0

    assert Nthreads > 1, 'Single process derivative calcs not supported'

    psf_grid, patch_grid = get_grids(patch_shape, psf_model.shape)

    # add regularization term to old ssqes
    Ndata = data.shape[0]
    eps_eff = eps / Ndata
    reg = np.sum((psf_model[:, 1:] - psf_model[:, :-1]) ** 2.)
    reg += np.sum((psf_model[1:, :] - psf_model[:-1, :]) ** 2.)
    reg *= eps_eff
    old_ssqes += reg 

    # heavy lifting, get derivatives
    derivatives = get_derivatives(data, dq, shifts, psf_model, old_ssqes,
                                  patch_shape, psf_grid, background, floor,
                                  gain, clip_parms, Nthreads, eps, h)

    # check that small step improves the model
    temp_psf = psf_model.copy() + derivatives * h
    ssqe, reg = evaluate((data, dq, shifts, temp_psf, psf_grid, background,
                          floor, gain, clip_parms, eps))
    assert (np.sum(ssqe) + reg - old_total_cost) < -small

    # possibly heavy, update the psf
    current_step = ini_step
    while True:
        temp_psf = psf_model.copy() + derivatives * current_step
        temp_psf = np.maximum(0.0, temp_psf)
        ssqe, reg = evaluate((data, dq, shifts, temp_psf, psf_grid, background,
                              floor, gain, clip_parms, eps))
        cost = np.sum(ssqe + reg)

        if ((old_cost - cost) / old_cost) < tol:
            return temp_psf, ssqe
        else:
            current_step /= rate

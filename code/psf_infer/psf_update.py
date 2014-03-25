import multiprocessing
import numpy as np

from .grid_definitions import get_grids
from .generation import render_psfs
from .patch_fitting import data_loss, evaluate, fit_single_patch
from .derivatives import get_derivatives, local_regularization

def update_psf(data, dq, psf_model, patch_shape, shifts,
               background, eps,
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

    # get current regularization term
    Ndata = data.shape[0]
    eps_eff = eps / Ndata
    old_reg = local_regularization(psf_model, eps_eff)

    # get current scaled squared error
    old_ssqe = evaluate((data, dq, shifts, psf_model, psf_grid, patch_shape,
                         background, floor, gain, clip_parms, loss_kind))

    # heavy lifting, get derivatives
    derivatives = get_derivatives(data, dq, shifts, psf_model, old_ssqe,
                                  old_reg, patch_shape, psf_grid, background,
                                  floor, gain, clip_parms, Nthreads, loss_kind,
                                  eps_eff, h)
    print derivatives
    assert 0
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
                              floor, gain, clip_parms, loss_kind, eps, True))
        cost = np.sum(ssqe + reg)

        if ((old_cost - cost) / old_cost) < tol:
            return temp_psf, ssqe
        else:
            current_step /= rate

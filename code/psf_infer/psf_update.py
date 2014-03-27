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
               tol=1.e-1, rate=0.5, ini_scale=1.0, Nsearch=32):
    """
    Update the psf model, using bfgs.
    """
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

    # check that small step improves the model
    temp_psf = psf_model.copy() - derivatives * h
    ssqe = evaluate((data, dq, shifts, temp_psf, psf_grid, patch_shape,
                     background, floor, gain, clip_parms, loss_kind))
    reg = local_regularization(psf_model, eps_eff)
    assert (np.sum(ssqe) + np.sum(reg) - old_total_cost) < -small

    # possibly heavy, update the psf
    old_cost = np.sum(ssqe) + np.sum(reg)
    current_scale = ini_scale
    regs = np.zeros(Nsearch)
    ssqes = np.zeros(Nsearch)
    costs = np.zeros(Nsearch)
    scales = np.zeros(Nsearch)
    for i in range(Nsearch):
        temp_psf = psf_model.copy() - derivatives * current_scale
        temp_psf = np.maximum(0.0, temp_psf)
        ssqe = evaluate((data, dq, shifts, temp_psf, psf_grid, patch_shape,
                         background, floor, gain, clip_parms, loss_kind))

        reg = np.sum(reg)
        ssqe = np.sum(ssqe)
        cost = ssqe + reg

        regs[i] = reg
        ssqes[i] = ssqe
        costs[i] = cost
        scales[i] = current_scale
        current_scale = np.exp(np.log(current_scale) - rate)

    ind = np.where(costs == np.min(costs))
    best_cost = costs[ind]
    best_scale = scales[ind]
    psf_model = psf_model - derivatives * best_scale
    psf_model = np.maximum(0.0, psf_model)

    return psf_model, best_cost, ssqes, regs, scales

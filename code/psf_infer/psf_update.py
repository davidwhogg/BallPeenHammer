import multiprocessing
import numpy as np

from .grid_definitions import get_grids
from .generation import render_psfs
from .patch_fitting import data_loss, evaluate, fit_single_patch
from .derivatives import get_derivatives, local_regularization

def update_psf(data, dq, psf_model, patch_shape, shifts,
               background, eps, loss_kind, floor, gain,
               clip_parms, Nthreads, h, q, small=1.e-10,
               tol=1.e-1, rate=0.25, ini_scale=0.5, Nsearch=64):
    """
    Update the psf model by calculating numerical derivatives and 
    finding the appropriate step in those directions.
    """
    psf_grid, patch_grid = get_grids(patch_shape, psf_model.shape)

    # get current regularization term
    Ndata = data.shape[0]
    old_reg = local_regularization(psf_model, eps)
    
    # get current scaled squared error
    old_ssqe = evaluate((data, dq, shifts, psf_model, psf_grid, patch_shape,
                         background, floor, gain, clip_parms, loss_kind, q))

    old_total_cost = np.sum(old_reg) + np.sum(old_ssqe)

    # heavy lifting, get derivatives
    derivatives = get_derivatives(data, dq, shifts, psf_model, old_ssqe,
                                  old_reg, patch_shape, psf_grid, background,
                                  floor, gain, clip_parms, Nthreads, loss_kind,
                                  eps, h, q)

    # check that small step improves the model
    temp_psf = psf_model.copy() - derivatives * h
    ssqe = evaluate((data, dq, shifts, temp_psf, psf_grid, patch_shape,
                     background, floor, gain, clip_parms, loss_kind, q))
    reg = local_regularization(temp_psf, eps)

    print np.sum(ssqe), np.sum(reg)
    assert (np.sum(ssqe) + np.sum(reg) - old_total_cost) < 0.0

    # find update to the psf
    current_scale = ini_scale
    regs = np.zeros(Nsearch)
    ssqes = np.zeros(Nsearch)
    costs = np.zeros(Nsearch)
    scales = np.zeros(Nsearch)
    for i in range(Nsearch):
        # perturb
        temp_psf = psf_model.copy() - derivatives * current_scale
        temp_psf = np.maximum(small, temp_psf)

        # evaluate
        ssqe = np.sum(evaluate((data, dq, shifts, temp_psf, psf_grid,
                                patch_shape, background, floor, gain,
                                clip_parms, loss_kind, q)))
        reg = np.sum(local_regularization(psf_model, eps))

        # store
        regs[i] = reg
        ssqes[i] = ssqe
        costs[i] = reg + ssqe
        scales[i] = current_scale

        # go down in scale
        current_scale = np.exp(np.log(current_scale) - rate)

    # look up best shift
    ind = np.where(costs == np.min(costs))
    best_cost = costs[ind]
    best_scale = scales[ind]

    # update
    psf_model = psf_model - derivatives * best_scale
    psf_model = np.maximum(small, psf_model)
    psf_model /= psf_model.max()

    return psf_model, best_cost, ssqes, regs, scales

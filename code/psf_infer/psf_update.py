import multiprocessing
import numpy as np

from .grid_definitions import get_grids
from .generation import render_psfs
from .patch_fitting import data_loss, evaluate, fit_single_patch
from .derivatives import get_derivatives, local_regularization

def update_psf(data, dq, psf_model, shifts, parms):
    """
    Update the psf model by calculating numerical derivatives and 
    finding the appropriate step in those directions.
    """
    # get current regularization term
    Ndata = data.shape[0]
    old_reg = local_regularization(psf_model, parms.eps)
    
    # get current scaled squared error
    old_ssqe = evaluate((data, dq, shifts, psf_model, parms, False))
    ind = old_ssqe < parms.max_ssqe
    old_total_cost = np.sum(old_reg) + np.mean(old_ssqe[ind])
    print 'Current ssqe:%0.2e, reg:%0.2e, total:%0.2e' % \
        (np.mean(old_ssqe[ind]), old_reg.sum(), old_total_cost)

    # heavy lifting, get derivatives
    derivatives = get_derivatives(data, dq, shifts, psf_model, old_ssqe,
                                  old_reg, parms)

    # check that small step improves the model
    temp_psf = psf_model.copy() - derivatives * parms.h
    ssqe = evaluate((data, dq, shifts, temp_psf, parms, False))
    ind = ssqe < parms.max_ssqe
    ssqe = np.mean(ssqe[ind])
    reg = local_regularization(temp_psf, parms.eps)
    assert (ssqe + np.sum(reg) - old_total_cost) < 0.0

    # find update to the psf
    current_scale = parms.search_scale
    regs = np.zeros(parms.Nsearch)
    ssqes = np.zeros(parms.Nsearch)
    costs = np.zeros(parms.Nsearch)
    scales = np.zeros(parms.Nsearch)
    for i in range(parms.Nsearch):
        # perturb
        temp_psf = psf_model.copy() - derivatives * current_scale
        temp_psf = np.maximum(parms.small, temp_psf)

        # evaluate
        ssqe = evaluate((data, dq, shifts, temp_psf, parms, False))
        ind = ssqe < parms.max_ssqe
        ssqe = np.mean(ssqe[ind])
        reg = np.sum(local_regularization(psf_model, parms.eps))

        # store
        regs[i] = reg
        ssqes[i] = ssqe
        costs[i] = reg + ssqe
        scales[i] = current_scale
        print i, ssqes[i], regs[i], costs[i], current_scale
        # go down in scale
        current_scale = np.exp(np.log(current_scale) - parms.search_rate)

    # look up best shift
    ind = np.where(costs == np.min(costs))
    best_reg = regs[ind]
    best_ssqe = ssqes[ind]
    best_cost = costs[ind]
    best_scale = scales[ind]

    # fill in broken searches
    ind = ssqe == np.inf
    idx = ssqe != np.inf
    ssqes[ind] = np.max(ssqes[idx])
    costs[ind] = np.max(costs[idx])

    print 'New ssqe:%0.2e, reg:%0.2e, total:%0.2e, at scale %0.2e' % \
        (best_ssqe, best_reg, best_cost, best_scale)
    
    # update
    psf_model = psf_model - derivatives * best_scale
    psf_model = np.maximum(parms.small, psf_model)
    psf_model /= psf_model.max()

    return psf_model, best_cost, ssqes, regs, scales

import multiprocessing
import numpy as np

from .patch_fitting import fit_single_patch, data_loss

def one_derivative(datum, dq, shift, psf_model, old_cost, psf_grid, patch_shape,
                   background, floor, gain, clip_parms, eps_eff, h):
    """
    Calculate the derivative for a single datum using forward differencing.
    """
    counts = np.zeros_like(psf_model)
    derivatives = np.zeros_like(psf_model)
    datum_shape = (datum.shape[0], patch_shape)
    for i in range(psf_model.shape[0]):
        for j in range(psf_model.shape[1]):
            temp_psf = psf_model.copy()
            temp_psf[i, j] += h

            new, reg = evaluate((data, dq, shifts, temp_psf, psf_grid,
                                 background, floor, gain, clip_parms, eps_eff))

            derivatives[i, j] = np.sum(new + reg - old_cost)

    ind = np.where(derivatives != 0.0)
    counts[ind] += 1.

    return counts, derivatives

def get_derivatives(data, dq, shifts, psf_model, old_costs, patch_shape,
                    psf_grid, background, floor, gain, clip_parms, Nthreads,
                    eps_eff, h):
    """
    Calculate the derivatives of the objective (in patch_fitting)
    with respect to the psf model.
    """
    assert ((data.shape.size == 2) & (dq.shape.size == 2)), \
        'Data should be the (un)raveled patch'
    assert Nthreads > 1, 'Single process derivative calcs not supported'

    # allocation
    coverage = np.ones_like(psf_model)
    derivatives = np.zeros_like(psf_model)
    
    # Map to the processes
    Ndata = data.shape[0]
    pool = multiprocessing.Pool(threads)
    mapfn = pool.map
    argslist = [None] * Ndata
    for i in range(Ndata):
        argslist[i] = (data[i], dq[i], shifts[i], psf_model, old_costs[i],
                       psf_grid, patch_shape, background, floor, gain,
                       clip_parms, eps_eff, h)
    results = list(mapfn(one_derivative, [args for args in argslist]))

    # Collect the results
    total_counts = np.zeros_like(psf_model)
    total_derivatives = np.zeros_like(psf_model)
    for i in range(data.shape[0]):
        total_counts += results[i][0]
        total_derivatives += results[i][1]

    # tidy up
    pool.close()
    pool.terminate()
    pool.join()

    return total_derivatives

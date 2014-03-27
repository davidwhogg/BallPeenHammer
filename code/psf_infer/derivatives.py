import multiprocessing
import numpy as np

from .patch_fitting import evaluate

def one_derivative((datum, dq, shift, psf_model, old_ssqe, old_reg, psf_grid,
                    patch_shape, background, floor, gain, clip_parms,
                    loss_kind, eps_eff, h)):
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

            new_ssqe = evaluate((datum, dq, shift, temp_psf, psf_grid,
                            patch_shape, background, floor, gain,
                            clip_parms, loss_kind))

            new_reg = local_regularization(temp_psf, eps_eff, idx=(i, j))

            derivatives[i, j] = np.sum(new_ssqe - old_ssqe)
            derivatives[i, j] += new_reg - old_reg[i, j]

    ind = np.where(derivatives != 0.0)
    counts[ind] += 1.

    return counts, derivatives

def get_derivatives(data, dq, shifts, psf_model, old_costs, old_reg,
                    patch_shape, psf_grid, background, floor, gain,
                    clip_parms, Nthreads, loss_kind, eps_eff, h):
    """
    Calculate the derivatives of the objective (in patch_fitting)
    with respect to the psf model.
    """
    assert (len(data.shape) == 2) & (len(dq.shape) == 2), \
        'Data should be the (un)raveled patch'
    assert Nthreads > 1, 'Single process derivative calcs not supported'

    # allocation
    coverage = np.ones_like(psf_model)
    derivatives = np.zeros_like(psf_model)
    
    # Map to the processes
    Ndata = data.shape[0]
    pool = multiprocessing.Pool(Nthreads)
    mapfn = pool.map
    argslist = [None] * Ndata
    for i in range(Ndata):
        argslist[i] = (data[None, i], dq[None, i], shifts[None, i], psf_model,
                       old_costs[i], old_reg, psf_grid, patch_shape,
                       background, floor, gain, clip_parms, loss_kind,
                       eps_eff, h)

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

    return total_derivatives / total_counts / h

def local_regularization(psf_model, eps, idx=None):
    """
    Calculate the local regularization for each pixel.
    """
    pm = np.array([-1, 1])
    psf_shape = psf_model.shape

    reg = np.zeros_like(psf_model)

    if idx is None:
        # axis 0
        idx = np.arange(psf_shape[0])
        ind = idx[:, None] + pm[None, :]
        ind[ind == -1] = 0 # boundary foo
        ind[ind == psf_shape[0]] = psf_shape[0] - 1 # boundary foo
        for i in range(psf_shape[0]):
            diff = psf_model[i, ind] - psf_model[i, idx][:, None]
            reg[i, :] += eps * np.sum(diff ** 2., axis=1)

        # axis 1
        idx = np.arange(psf_shape[1])
        ind = idx[:, None] + pm[None, :]
        ind[ind == -1] = 0 # boundary foo
        ind[ind == psf_shape[1]] = psf_shape[1] - 1 # boundary foo
        for i in range(psf_shape[1]):
            diff = psf_model[ind, i] - psf_model[idx, i][:, None]
            reg[:, i] += eps * np.sum(diff ** 2., axis=1)

    else:
        idx = np.array(idx)
        ind = idx[:, None] + pm[None, :]
        ind[ind == -1] = 0
        ind[ind == psf_shape[0]] = psf_shape[0] - 1 # assumes square psf model

        value = psf_model[idx[0], idx[1]]
        reg = eps * np.sum((psf_model[ind[0], idx[1]] - value) ** 2.)
        reg += eps * np.sum((psf_model[idx[0], ind[1]] - value) ** 2.)

    return reg

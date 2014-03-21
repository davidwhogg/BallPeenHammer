import numpy as np

from .generation import render_psfs
from .grid_definitions import get_grids

def fit_single_patch((data, psf, flat, dq, background, floor, gain,
                      clip_parms)):
    """
    Fit a single patch, return the scale for the psf plus any
    background parameters.  Takes in flattened arrays for
    data and psf.
    """
    # not sure why this `* flat` nonsense is the only 
    # mode that works!
    if background is None:
        A = np.atleast_2d(psf * flat).T
    elif background == 'constant':
        A = np.vstack((psf * flat, np.ones_like(psf) * flat)).T
    elif background == 'linear':
        N = np.sqrt(psf.size).astype(np.int)
        x, y = np.meshgrid(range(N), range(N))
        A = np.vstack((psf * flat, np.ones_like(psf) * flat,
                       x.ravel() * flat, y.ravel() * flat)).T
    else:
        assert False, 'Background model not supported: %s' % background

    ind = np.where(dq == 0)[0]

    # fit the data using least squares
    rh = np.dot(A[ind, :].T, data[ind])
    try:
        lh = np.linalg.inv(np.dot(A[ind, :].T, A[ind, :]))
        parms = np.dot(lh, rh)
    except:
        parms = np.zeros(A.shape[1])

    bkg = make_background(data, A, parms, background)

    # sigma clip if desired
    if (clip_parms is not None) & (np.any(parms != 0)):
        Niter = clip_parms[0]
        sigma = clip_parms[1]
        for i in range(Niter):
            # define model and noise
            model = psf[ind] * parms[0] + bkg[ind]
            var = floor + gain * np.abs(model)
            
            # sigma clip
            chi = np.abs(data[ind] - model) / np.sqrt(var)
            if type(sigma) != float:
                condition = chi - sigma[ind]
            else:
                condition = chi - sigma
            idx = np.where(condition > 0)[0]
            ind = np.delete(ind, idx)

            # refit
            rh = np.dot(A[ind, :].T, data[ind])
            lh = np.linalg.inv(np.dot(A[ind, :].T, A[ind, :]))
            parms = np.dot(lh, rh)
            bkg = make_background(data, A, parms, background)

        return parms[0], parms[1:], bkg, ind
    else:
        return parms[0], None, bkg, ind

def make_background(data, A, parms, background):
    """
    Make the backgound model for a patch
    """
    bkg = np.zeros_like(data)
    if background is not None:
        for i in range(A.shape[1] - 1):
            bkg += A[:, i + 1] * parms[i + 1]
    return bkg

def data_loss(data, model, kind, floor, gain, var=None, q=0.02):
    """
    Return the specified error/loss/nll.
    """
    if np.all(model == 0):
        sqe = np.Inf
    else:
        sqe = (data - model) ** 2.  

    if kind == 'sqe':
        return sqe
    if kind == 'ssqe-data':
        ssqe = sqe / data ** 2.
    if kind == 'ssqe-model':
        ssqe = sqe / model ** 2.
    if kind == 'ssqe-sum-data':
        ssqe = sqe / np.sum(data) ** 2.
    if kind == 'ssqe-sum-model':
        ssqe = sqe / np.sum(model) ** 2.
    if kind == 'nll-model':
        var = floor + gain * np.abs(model)
        ssqe = 0.5 * (np.log(var) + sqe / var)
    return ssqe

def render_models(data, dq, psf_model, shifts, floor, gain, clip_parms=None,
                  background='constant', loss_kind='nll-model'):
    """
    Render a set of models
    """
    loss_kind = 'sqe'
    patch_shape = (data.shape[1], data.shape[2])
    psf_grid, patch_grid = get_grids(patch_shape, psf_model.shape)

    rendered_psfs = render_psfs(psf_model, shifts, data.shape, psf_grid[0],
                                psf_grid[1])
    xpg, ypg = patch_grid[0], patch_grid[1]

    ssqe = np.zeros(data.shape[0])
    models = np.zeros((data.shape[0], data.shape[1] * data.shape[2]))

    for i in range(data.shape[0]):
        datum = data[i].ravel()
        psf = rendered_psfs[i].ravel()
        flat = np.ones_like(datum) # fix me!!!
        flux, bkg_parms, bkg, ind = fit_single_patch((datum,
                                                      psf, flat,
                                                      dq[i].ravel(),
                                                      background, floor,
                                                      gain, clip_parms))
        model = flat * (flux * psf + bkg)
        s = data_loss(datum[ind], model[ind], loss_kind, floor, gain)
        ssqe[i] = np.sum(s)
        models[i] = model

    return models, ssqe

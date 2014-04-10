import numpy as np

from .generation import render_psfs
from .grid_definitions import get_grids
from scipy.ndimage.morphology import binary_dilation as grow_mask

def evaluate((data, dq, shifts, psf_model, parms, core)):
    """
    Compute the scaled squared error and regularization under the current 
    model.
    """
    if core:
        patch_shape = parms.core_shape
    else:
        patch_shape = parms.patch_shape

    psfs = render_psfs(psf_model, shifts, patch_shape, parms.psf_grid)
    ssqe = np.zeros_like(data)
    for i in range(data.shape[0]):
        flux, bkg_parms, bkg, ind = fit_single_patch((data[i], psfs[i],
                                                      dq[i], parms))
        model = flux * psfs[i] + bkg

        # chi-squared like term
        ssqe[i, ind] = data_loss(data[i][ind], model[ind], bkg[ind], parms)

    return ssqe

def fit_single_patch((data, psf, dq, parms)):
    """
    Fit a single patch, return the scale for the psf plus any
    background parameters.  Takes in flattened arrays for
    data and psf.
    """
    gain = parms.gain
    floor = parms.floor
    clip_parms = parms.clip_parms
    background = parms.background

    if background is None:
        A = np.atleast_2d(psf).T
    elif background == 'constant':
        A = np.vstack((psf, np.ones_like(psf))).T
    elif background == 'linear':
        N = np.sqrt(psf.size).astype(np.int)
        x, y = np.meshgrid(range(N), range(N))
        A = np.vstack((psf, np.ones_like(psf),
                       x.ravel(), y.ravel())).T
    else:
        assert False, 'Background model not supported: %s' % background

    ind = dq == 0

    # fit the data using least squares
    rh = np.dot(A[ind, :].T, data[ind])
    try:
        lh = np.linalg.inv(np.dot(A[ind, :].T, A[ind, :]))
        fit_parms = np.dot(lh, rh)
    except:
        fit_parms = np.zeros(A.shape[1])

    bkg = make_background(data, A, fit_parms, background)

    # sigma clip if desired
    if (clip_parms is not None) & (np.any(parms != 0)):
        Niter = clip_parms[0]
        tol = clip_parms[1]
        for i in range(Niter):
            # define model and noise
            scaled_psf = psf[ind] * fit_parms[0]
            model = scaled_psf + bkg[ind]
            var = floor + gain * np.abs(model) + (q * scaled_psf) ** 2.
            
            # sigma clip
            chi = np.zeros_like(data)
            chi[ind] = np.abs(data[ind] - model) / np.sqrt(var)
            if type(tol) != float:
                condition = chi - tol[ind]
            else:
                condition = chi - tol
            
            # redefine mask, grow and add to dq mask.
            ind = 1 - ind.reshape(25,25)
            idx = grow_mask((condition > 0).reshape(25,25))
            ind = ind | idx
            ind = (1 - ind).ravel()

            # refit
            rh = np.dot(A[ind, :].T, data[ind])
            lh = np.linalg.inv(np.dot(A[ind, :].T, A[ind, :]))
            fit_parms = np.dot(lh, rh)
            bkg = make_background(data, A, fit_parms, background)

        return fit_parms[0], fit_parms[1:], bkg, ind
    else:
        return fit_parms[0], None, bkg, ind

def make_background(data, A, fit_parms, background):
    """
    Make the backgound model for a patch
    """
    bkg = np.zeros_like(data)
    if background is not None:
        for i in range(A.shape[1] - 1):
            bkg += A[:, i + 1] * fit_parms[i + 1]
    return bkg

def data_loss(data, model, bkg, parms):
    """
    Return the specified error/loss/nll.
    """
    kind = parms.loss_kind

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
        var = parms.floor + parms.gain * np.abs(model) + \
            (parms.q * (model - bkg)) ** 2.
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
    if floor is None:
        var = 1.
    else:
        var = floor + gain * np.abs(model)
    pl.imshow(((data - model) ** 2. / var).reshape(patch_shape),
              interpolation='nearest', origin='lower')
    pl.colorbar()
    f.savefig('../plots/foo.png')
    assert 0

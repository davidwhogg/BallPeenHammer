import numpy as np

from .generation import render_psfs

def PatchFitter(data, ini_psf, ini_flat, background='linear',
                sequence=['shifts', 'flat', 'psf'], tol=1.e-4):
    """
    Patch fitting routines for BallPeenHammer.
    """
    assert 'shifts' not in sequence, 'shift inference not yet supported'
    assert 'flat' in sequence, 'fitting for flat, should be in sequence'
    assert background in [None, 'constant', 'linear']

    current_psf = ini_psf
    current_loss = np.Inf
    current_flat = ini_flat

    rendered_psfs = None

    while True:
        for kind in sequence:
            if kind == 'flat':
                current_flat = update_flat(data, current_flat, rendered_psfs,
                                           background)

def update_flat(data, current_flat, rendered_psfs, detector_grid,
                patch_centers, psf_model=None, shifts=None,
                psf_grid=None, tol=1.e-4):
    """
    Update the flat field given current psf model
    """
    if rendered_psf.shape != data.shape:
        assert shifts is not None
        assert psf_grid is not None
        assert psf_model is not None
        rendered_psfs = render_psfs(psf_model, shifts, data.shape, psf_grid[0],
                                    psf_grid[1])

    # shorthand
    xdg, ydg = detector_grid[0], detector_grid[1]
    xpc, ypc = patch_centers[0], patch_centers[1]

    # infer flat
    while True:
        new_tse = 0.0
        norm = np.zeros_like(current_flat)
        new_flat = np.zeros_like(current_flat)
        for i in range(data.shape[0]):
            cf = current_flat[xdg + xpc[i], ydg + ypc[i]]
            amp, bkg_parms = fit_single_patch(data[i].ravel() / cf.ravel(), 
                                              psf.ravel(), background)
            model = psf * amp + bkg.reshape(data[i].shape[0],
                                            data[i].shape[1])
            model *= cf

            se = np.sum((data[i] - model) ** 2.)

            norm[xdg + xpc[i], ydg + ypc[i]] += 1
            new_flat[xdg + xpc[i], ydg + ypc[i]] += (data[i] / model)
            new_tst += se
            
        new_flat /= norm
        new_flat *= new_flat.size / new_flat.sum()

        # assess convergence
        if (tse - new_tse) / new_tse > tol:
            tse = new_tse
            current_flat = new_flat
        else:
            return new_flat, norm, new_tse

def fit_single_patch(data, psf, background):
    """
    Fit a single patch, return the scale for the psf plus any
    background parameters.  Takes in flattened values for
    data and psf.
    """
    if background is None:
        A = np.atleast_2d(psf)
    elif background is 'constant':
        A = np.vstack((psf, np.ones_like(psf)))
    elif background is 'linear':
        A = np.vstack((psf, np.ones_like(psf),
                       np.arange(psf.size), np.arange(psf.size)))
    else:
        assert False, 'Background model not supported.'

    rh = np.dot(A.T, data)
    lh = np.linalg.inv(np.dot(A.T, A))
    
    parms = np.dot(lh, rh)
    bkg = np.zeros_like(data)
    for i in range(A.shape[0] - 1):
        bkg += A[i + 1] * parms[i + 1]

    return parms[0], parms[1:], bkg

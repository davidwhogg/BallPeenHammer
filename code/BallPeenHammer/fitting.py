import numpy as np
import matplotlib.pyplot as pl

from .generation import render_psfs

def PatchFitter(data, ini_psf, ini_flat, detector_grid, psf_grid,
                patch_centers, background='linear',
                sequence=['shifts', 'flat', 'psf'], tol=1.e-4,
                shifts=None):
    """
    Patch fitting routines for BallPeenHammer.
    """
    assert 'shifts' not in sequence, 'shift inference not yet supported'
    assert 'flat' in sequence, 'fitting for flat, should be in sequence'
    assert background in [None, 'constant', 'linear']

    current_psf = ini_psf.copy()
    current_loss = np.Inf
    current_flat = ini_flat.copy()

    rendered_psfs = None

    while True:
        for kind in sequence:
            if kind == 'flat':
                current_flat, counts, sqe = update_flat(data, current_flat,
                                                        rendered_psfs,
                                                        detector_grid,
                                                        patch_centers,
                                                        background, 
                                                        psf_model=current_psf,
                                                        shifts=shifts,
                                                        psf_grid=psf_grid)

        
        return current_flat, current_psf

def update_flat(data, current_flat, rendered_psfs, detector_grid,
                patch_centers, background, psf_model=None, shifts=None,
                psf_grid=None, tol=1.e-4):
    """
    Update the flat field given current psf model
    """
    if rendered_psfs is None:
        assert shifts is not None
        assert psf_grid is not None
        assert psf_model is not None

        rendered_psfs = render_psfs(psf_model, shifts, data.shape, psf_grid[0],
                                    psf_grid[1])

    # shorthand
    xdg, ydg = detector_grid[0], detector_grid[1]
    xpc, ypc = patch_centers[0], patch_centers[1]

    # infer flat
    for dummy in range(3):
        sqe = 0.0
        norm = np.zeros_like(current_flat)
        new_flat = np.zeros_like(current_flat)
        for i in range(data.shape[0]):
            cf = current_flat[xdg + xpc[i], ydg + ypc[i]].copy()
            psf = rendered_psfs[i].copy()
            amp, bkg_parms, bkg = fit_single_patch(data[i].ravel().copy() / 
                                                   cf.ravel(), psf.ravel(),
                                                   background)
            model = psf * amp * cf
            sqe += np.sum((data[i].copy() - model) ** 2.)

            norm[xdg + xpc[i], ydg + ypc[i]] += 1.
            new_flat[xdg + xpc[i], ydg + ypc[i]] += (data[i] / model) - cf
    
        print sqe
        new_flat /= norm
        #new_flat *= new_flat.size / new_flat.sum()
        current_flat += new_flat.copy()

    return new_flat, norm, sqe

def fit_single_patch(data, psf, background):
    """
    Fit a single patch, return the scale for the psf plus any
    background parameters.  Takes in flattened values for
    data and psf.
    """
    if background is None:
        A = np.atleast_2d(psf).T
    elif background is 'constant':
        assert 0, 'this or transpose???'
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

    if background is not None:
        for i in range(A.shape[0] - 1):
            bkg += A[:, i + 1] * parms[i + 1]
        
        return parms[0], parms[1:], bkg
    else:
        return parms[0], None, bkg

"""

    tse = 1.e200
    count = 0
    while True:
        new_tse = 0.0
        print count, tse, new_tse
        #norm = np.zeros_like(current_flat)
        #new_flat = np.zeros_like(current_flat)
        count += 1.
        for i in range(data.shape[0]):
            # shorthand
            cf = current_flat[xdg + xpc[i], ydg + ypc[i]].copy()
            #psf = rendered_psfs[i]
            #cal_data = data[i].copy().ravel() / cf.ravel()

            # fit the model
            #amp, bkg_parms, bkg = fit_single_patch(data[i].ravel().copy() * current_flat[xdg + xpc[i], ydg + ypc[i]].ravel().copy(), rendered_psfs[i].ravel().copy(), background)
            model = rendered_psfs[i].copy() * count #+ bkg.reshape(data[i].shape[0],
                                            #data[i].shape[1])
            #model *= cf
            se = (data[i].copy() - model) ** 2.
            #norm[xdg + xpc[i], ydg + ypc[i]] += 1
            #new_flat[xdg + xpc[i], ydg + ypc[i]] += (data[i] / model)
            new_tse += se.sum()
        print count, current_flat[xdg + xpc[i], ydg + ypc[i]].ravel()
        #count = count + 1
        # scale, sum to one.
        #new_flat /= norm
        #new_flat *= new_flat.size / new_flat.sum()

        print 'end', count, tse, new_tse
        print 
        tse = 1. * new_tse
        #print (tse - new_tse) / new_tse
        # assess convergence
        #if (tse - new_tse) / new_tse > tol:
            #tse = new_tse.copy()
            #current_flat = new_flat
            #print tse, new_tse
            #print (tse - new_tse) / new_tse
        #    print 
        #else:
        #    return new_flat, norm, new_tse

"""

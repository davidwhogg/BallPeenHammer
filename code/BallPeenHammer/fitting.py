import multiprocessing
import numpy as np

from .generation import render_psfs

from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b, fmin_tnc, fmin

def PatchFitter(data, dq, ini_psf, ini_flat, patch_grid, psf_grid,
                patch_centers, background='linear',
                sequence=['shifts', 'flat', 'psf'], tol=1.e-4, eps=1.e-2,
                ini_shifts=None, threads=1):
    """
    Patch fitting routines for BallPeenHammer.
    """
    assert background in [None, 'constant', 'linear']

    current_psf = ini_psf.copy()
    current_loss = np.Inf
    current_flat = ini_flat.copy()
    if ini_shifts is not None:
        shifts = ini_shifts.copy()

    rendered_psfs = None

    while True:
        for kind in sequence:
            if kind == 'shifts':
                shifts, ssqe = update_shifts(data, dq, current_flat, current_psf,
                                            psf_grid, patch_grid, patch_centers,
                                            shifts, background, threads)
            if kind == 'psf':
                current_psf = update_psf(data, current_flat, current_psf, 
                                         psf_grid, patch_grid, patch_centers,
                                         shifts, background, eps, threads)
            if kind == 'flat':
                current_flat, counts, sqe = update_flat(data, current_flat,
                                                        rendered_psfs,
                                                        patch_grid,
                                                        patch_centers,
                                                        background, 
                                                        psf_model=current_psf,
                                                        shifts=shifts,
                                                        psf_grid=psf_grid,
                                                        tol=tol)

        
        return current_flat, current_psf

def update_shifts(data, dq, current_flat, current_psf, psf_grid, patch_grid, 
                  patch_centers, shifts, background, threads):
    """
    Update the estimate of the subpixel shifts, given current psf and flat.
    """
    xpg, ypg = patch_grid[0], patch_grid[1]
    xpc, ypc = patch_centers[0], patch_centers[1]

    ssqe = np.zeros(data.shape[0])
    if threads == 1:
        for i in range(data.shape[0]):
            p0 = shifts[i]
            flat = current_flat[xpg + xpc[i], ypg + ypc[i]]

            
            # fmin seems to perform better than fmin_bfgs or fmin_l_bfgs_b
            res = fmin(shift_loss, p0, full_output=True, disp=False,
                       args=(current_psf, data[i], dq[i], flat, psf_grid, 
                             background))
            shifts[i] = res[0]
            ssqe[i] = res[1]
    else:
        mapfn = multiprocessing.Pool(threads).map
        argslist = [None] * data.shape[0]
        for i in range(data.shape[0]):
            flat = current_flat[xpg + xpc[i], ypg + ypc[i]]
            argslist[i] = [shifts[i], current_psf, data[i], dq[i], flat, psf_grid,
                           background]

        results = list(mapfn(update_single_shift, [args for args in argslist]))
        for i in range(data.shape[0]):
            shifts[i] = results[i][0]
            ssqe[i] = results[i][1]

    return shifts, ssqe

def update_single_shift(*args):
    """
    Update a single shift, enabling multiprocessing
    """
    args = args[0]
    p0 = args[0]
    current_psf = args[1]
    datum = args[2]
    dq = args[3]
    flat = args[4]
    psf_grid = args[5]
    background = args[6]
    bounds = [(-0.5, 0.5), (-0.5, 0.5)]

    # fmin seems to perform better than fmin_bfgs or fmin_l_bfgs_b
    res = fmin(shift_loss, p0, full_output=True, disp=False,
               args=(current_psf, datum, dq, flat, psf_grid, 
                     background))
    return res

def shift_loss(shift, psf_model, data, dq, flat, psf_grid, background):
    """
    Evaluate the shift for a given patch.
    """
    # Horrible hack for minimizers w/o bounds
    if (np.abs(shift[0]) > 0.5) | (np.abs(shift[1]) > 0.5):
        return 1.e10

    shape = (1, data.shape[0], data.shape[1])
    psf = render_psfs(psf_model, np.array([shift]), shape, psf_grid[0],
                      psf_grid[1])
    ind = dq == 0
    flux, bkg_parms, bkg = fit_single_patch(data.ravel(),
                                            psf.ravel(),
                                            flat.ravel(),
                                            ind.ravel(),
                                            background)
    model = flat * (flux * psf[0] + bkg.reshape(data.shape))
    ssqe = np.sum((data[ind] - model[ind]) ** 2. / model[ind] ** 2.)
    return ssqe    

def update_psf(data, current_flat, current_psf, psf_grid, patch_grid, 
               patch_centers, shifts, background, eps):
    """
    Update the psf model, using bfgs.
    """
    assert 0, 'Update: multiprocessing, dq'
    p0 = current_psf.ravel().copy()
    return fmin_bfgs(psf_loss, p0, 
                     args=(data, current_flat, psf_grid, patch_grid, patch_centers, 
                           shifts, background, eps))


def psf_loss(psf_model, data, current_flat, psf_grid, patch_grid,
             patch_centers, shifts, background, eps):
    """
    Evaluate the current psf model, given a flat and shifts.
    """
    D = np.sqrt(psf_model.size)
    psf_model = psf_model.reshape(D, D)
    rendered_psfs = render_psfs(psf_model, shifts, data.shape, psf_grid[0],
                                psf_grid[1])

    patch_shape = (data.shape[1], data.shape[2])
    xpg, ypg = patch_grid[0], patch_grid[1]
    xpc, ypc = patch_centers[0], patch_centers[1]

    # compute squared error
    sqe = 0.0
    for i in range(data.shape[0]):
        psf = rendered_psfs[i]
        flat = current_flat[xpg + xpc[i], ypg + ypc[i]]
        flux, bkg_parms, bkg = fit_single_patch(data[i].ravel(),
                                                psf.ravel(),
                                                flat.ravel(),
                                                background)
        model = flat * (flux * psf + bkg.reshape(patch_shape))
        sqe += np.sum((data[i] - model) ** 2.)

    # regularization
    sqg = np.gradient(psf_model)
    sqg = np.append(sqg[0].ravel(), sqg[1].ravel())
    sqg = sqg ** 2.
    reg = np.sum(eps * sqg / sqg.mean())
    
    print 'SQE = %0.3f, REG = %0.3f' % (sqe, reg)

    return sqe + reg

def update_flat(data, current_flat, rendered_psfs, patch_grid,
                patch_centers, background, psf_model=None, shifts=None,
                psf_grid=None, tol=1.e-3):
    """
    Update the flat field given current psf model
    """
    assert 0, 'Update: multithreading, dq'
    if rendered_psfs is None:
        assert shifts is not None
        assert psf_grid is not None
        assert psf_model is not None

        rendered_psfs = render_psfs(psf_model, shifts, data.shape, psf_grid[0],
                                    psf_grid[1])

    # shorthand
    patch_shape = (data.shape[1], data.shape[2])
    xpg, ypg = patch_grid[0], patch_grid[1]
    xpc, ypc = patch_centers[0], patch_centers[1]
    
    # infer flat
    tse = np.Inf
    while True:
        new_tse = 0.0
        norm = np.zeros_like(current_flat)
        new_flat = np.zeros_like(current_flat)
        for i in range(data.shape[0]):
            psf = rendered_psfs[i]
            flat = current_flat[xpg + xpc[i], ypg + ypc[i]]
            flux, bkg_parms, bkg = fit_single_patch(data[i].ravel(),
                                                    psf.ravel(),
                                                    flat.ravel(),
                                                    background)
            model = flat * (flux * psf + bkg.reshape(patch_shape))
            se = (data[i] - model) ** 2.
            norm[xpg + xpc[i], ypg + ypc[i]] += 1.
            new_flat[xpg + xpc[i], ypg + ypc[i]] += (data[i] / model) * \
                current_flat[xpg + xpc[i], ypg + ypc[i]]
            new_tse += se.sum()

        new_flat /= norm
        new_flat *= new_flat.size / new_flat.sum()

        print tse, new_tse
        if (tse - new_tse) / new_tse > tol:
            tse = new_tse
            current_flat = new_flat
        else:
            return new_flat, norm, new_tse

def fit_single_patch(data, psf, flat, ind, background):
    """
    Fit a single patch, return the scale for the psf plus any
    background parameters.  Takes in flattened values for
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

    # fit the data using least squares
    rh = np.dot(A[ind, :].T, data[ind])
    try:
        lh = np.linalg.inv(np.dot(A[ind, :].T, A[ind, :]))
        parms = np.dot(lh, rh)
    except:
        parms = np.zeros(A.shape[1])

    # make background model
    bkg = np.zeros_like(data)
    if background is not None:
        for i in range(A.shape[1] - 1):
            bkg += A[:, i + 1] * parms[i + 1]
        
        return parms[0], parms[1:], bkg
    else:
        return parms[0], None, bkg

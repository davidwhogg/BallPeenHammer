import multiprocessing
import numpy as np
import pyfits as pf

from .generation import render_psfs

from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b, fmin_tnc, fmin
from scipy.optimize import fmin_cg, fmin_powell

def PatchFitter(data, dq, ini_psf, ini_flat, patch_grid, psf_grid,
                patch_centers, background='linear', rendered_psfs=None,
                sequence=['shifts', 'flat', 'psf'], tol=1.e-4, eps=1.e-4,
                ini_shifts=None, shift_threads=1, psf_threads=1, floor=None,
                gain=None, maxiter=np.Inf, dumpfilebase=None, trim_frac=0.005,
                min_frac=0.75, loss_kind='ssqe-sum-data'):
    """
    Patch fitting routines for BallPeenHammer.
    """
    assert background in [None, 'constant', 'linear']

    current_psf = ini_psf.copy()
    current_loss = np.Inf
    current_flat = ini_flat.copy()
    if ini_shifts is not None:
        shifts = ini_shifts.copy()

    Nmin = np.ceil(min_frac * data.shape[0]).astype(np.int)
    mask = np.arange(data.shape[0]).astype(np.int)

    tot_ssqe = np.Inf
    iterations = 0
    while True:
        if iterations > maxiter:
            return current_flat, current_psf, shifts
        for kind in sequence:
            if kind == 'shifts':
                shifts, ssqe = update_shifts(data[mask], dq[mask], current_flat,
                                             current_psf, psf_grid, patch_grid,
                                             patch_centers, shifts,
                                             background, shift_threads,
                                             loss_kind, floor, gain)

                print 'Shift step done, pre ssqe: ', ssqe.sum()
                if (trim_frac is not None) & (mask.size > Nmin):
                    Ntrim = np.ceil(mask.size * trim_frac).astype(np.int)
                    if (mask.size - Ntrim < Nmin):
                        Ntrim = mask.size - Nmin
                    ind = np.argsort(ssqe)[:-Ntrim]
                    mask = mask[ind]
                    shifts = shifts[ind]
                    shifts, ssqe = update_shifts(data[mask], dq[mask],
                                                 current_flat, current_psf,
                                                 psf_grid, patch_grid,
                                                 patch_centers, shifts,
                                                 background, shift_threads,
                                                 loss_kind, floor, gain)

                print 'Shift step done, post ssqe: ', ssqe.sum()
                if dumpfilebase is not None:
                    f = open(dumpfilebase + '_mask_%d.dat' % iterations, 'w')
                    for ind in mask:
                        f.write('%d\n' % ind)
                    f.close()
                    np.savetxt(dumpfilebase + '_shifts_%d.dat' % iterations,
                               shifts)
                    np.savetxt(dumpfilebase + '_shift_ssqe_%d.dat' % iterations,
                               ssqe)

            if kind == 'psf':
                current_psf, ssqe = update_psf(data[mask], dq[mask],
                                               current_flat, current_psf,
                                               psf_grid, patch_grid,
                                               patch_centers, shifts,
                                               background, eps, psf_threads,
                                               loss_kind, floor, gain)

                print 'Psf step done, ssqe: ', ssqe.sum()
                if dumpfilebase is not None:
                    h = pf.PrimaryHDU(current_psf)
                    h.writeto(dumpfilebase + '_psf_%d.fits' % iterations,
                              clobber=True)
                    np.savetxt(dumpfilebase + '_psf_ssqe_%d.dat' % iterations,
                               ssqe)

            if kind == 'flat':
                assert False, 'need to update to current loss functions'
                current_flat, counts, ssqe = update_flat(data, current_flat,
                                                         rendered_psfs,
                                                         patch_grid,
                                                         patch_centers,
                                                         background, 
                                                         psf_model=current_psf,
                                                         shifts=shifts,
                                                         psf_grid=psf_grid,
                                                         tol=tol)
                print 'Flat step done, ssqe: ', ssqe.sum()

        if ((tot_ssqe - np.sum(ssqe)) / np.sum(ssqe)) < tol:
            return current_flat, current_psf, shifts
        else:
            iterations += 1
            tot_ssqe = np.sum(ssqe)

def update_shifts(data, dq, current_flat, current_psf, psf_grid, patch_grid, 
                  patch_centers, shifts, background, threads, loss_kind,
                  floor, gain):
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
            res = update_single_shift((p0, current_psf, data[i], dq[i], 
                                       flat, psf_grid, background, loss_kind,
                                       floor, gain))
            
            shifts[i] = res[0]
            ssqe[i] = res[1]
    else:
        pool = multiprocessing.Pool(threads)
        mapfn = pool.map
        argslist = [None] * data.shape[0]
        for i in range(data.shape[0]):
            flat = current_flat[xpg + xpc[i], ypg + ypc[i]]
            argslist[i] = [shifts[i], current_psf, data[i], dq[i], flat,
                           psf_grid, background, loss_kind, floor, gain]

        results = list(mapfn(update_single_shift, [args for args in argslist]))
        for i in range(data.shape[0]):
            shifts[i] = results[i][0]
            ssqe[i] = results[i][1]

        pool.close()
        pool.terminate()
        pool.join()

    return shifts, ssqe / data.shape[0]

def update_single_shift((p0, current_psf, datum, dq, flat, psf_grid,
                         background, loss_kind, floor, gain)):
    """
    Update a single shift, enabling multiprocessing
    """
    # fmin seems to perform better than fmin_bfgs or fmin_l_bfgs_b
    # need to try fmin_powell
    res = fmin(shift_loss, p0, full_output=True, disp=False,
               args=(current_psf, datum, dq, flat, psf_grid, 
                     background, loss_kind, floor, gain))
    return res

def shift_loss(shift, psf_model, data, dq, flat, psf_grid, background,
               loss_kind, floor, gain):
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
    flux, bkg_parms, bkg = fit_single_patch((data.ravel(), psf.ravel(),
                                             flat.ravel(), ind.ravel(),
                                             background))
    model = flat * (flux * psf[0] + bkg.reshape(data.shape))
    ssqe = data_loss(data[ind], model[ind], loss_kind, floor, gain)

    return np.sum(ssqe)

def update_psf(data, dq, current_flat, current_psf, psf_grid, patch_grid, 
               patch_centers, shifts, background, eps, threads, loss_kind,
               floor, gain):
    """
    Update the psf model, using bfgs.
    """
    global count
    count = 0

    p0 = np.log(current_psf.ravel().copy())

    # powell is most efficient w/o gradients, just do one iteration
    res = fmin_powell(psf_loss, p0, maxiter=1,
                      args=(data, dq, current_flat, psf_grid, patch_grid,
                            patch_centers, shifts, background, eps, threads,
                            loss_kind, floor, gain))

    # get ssqe vector
    ssqe = psf_loss(res, data, dq, current_flat, psf_grid, patch_grid,
                    patch_centers, shifts, background, eps, threads, loss_kind,
                    floor, gain, summation=False)

    res = np.exp(res.reshape(current_psf.shape[0], current_psf.shape[1]))
    return res, ssqe

def psf_loss(psf_model, data, dq, current_flat, psf_grid, patch_grid,
             patch_centers, shifts, background, eps, threads, loss_kind, 
             floor, gain, summation=True):
    """
    Evaluate the current psf model, given a flat and shifts.
    """
    global count

    D = np.sqrt(psf_model.size)
    psf_model = np.exp(psf_model.reshape(D, D))
    rendered_psfs = render_psfs(psf_model, shifts, data.shape, psf_grid[0],
                                psf_grid[1])

    patch_shape = (data.shape[1], data.shape[2])
    xpg, ypg = patch_grid[0], patch_grid[1]
    xpc, ypc = patch_centers[0], patch_centers[1]

    # compute squared error
    # note threads=1 is faster depending on data amount?
    ssqe = np.zeros(data.shape[0])
    if threads == 1:
        for i in range(data.shape[0]):
            ind = dq[i] == 0
            psf = rendered_psfs[i]
            flat = current_flat[xpg + xpc[i], ypg + ypc[i]]
            flux, bkg_parms, bkg = fit_single_patch((data[i].ravel(),
                                                     psf.ravel(), flat.ravel(),
                                                     ind.ravel(), background))
            model = flat * (flux * psf + bkg.reshape(patch_shape))
            ssqe[i] = np.sum(data_loss(data[i, ind], model[ind], loss_kind,
                                       floor, gain))
    else:
        pool = multiprocessing.Pool(threads)
        mapfn = pool.map
        argslist = [None] * data.shape[0]
        for i in range(data.shape[0]):
            ind = dq[i] == 0
            psf = rendered_psfs[i]
            flat = current_flat[xpg + xpc[i], ypg + ypc[i]]
            argslist[i] = [data[i].ravel(), psf.ravel(), flat.ravel(),
                           ind.ravel(), background]

        results = list(mapfn(fit_single_patch, [args for args in argslist]))
        for i in range(data.shape[0]):
            ind = dq[i] == 0
            psf = rendered_psfs[i]
            flat = current_flat[xpg + xpc[i], ypg + ypc[i]]
            flux = results[i][0]
            bkg_parms = results[i][1]
            bkg = results[i][2]
            model = flat * (flux * psf + bkg.reshape(patch_shape))
            ssqe[i] = np.sum(data_loss(data[i, ind], model[ind], loss_kind,
                                       floor, gain))

        pool.close()
        pool.terminate()
        pool.join()

    # scale by number of data points
    ssqe /= data.shape[0]

    if summation:
        # regularization
        reg = np.sum((psf_model[:, 1:] - psf_model[:, :-1]) ** 2.)
        reg += np.sum((psf_model[1:, :] - psf_model[:-1, :]) ** 2.)
        reg *= eps

        if np.mod(count, 50) == 0:
            print count, 'SSQE = %0.6f, REG = %0.6f' % (ssqe.sum(), reg)
        count += 1

        return np.sum(ssqe) + reg
    else:
        return ssqe

def update_flat(data, current_flat, rendered_psfs, patch_grid,
                patch_centers, background, psf_model=None, shifts=None,
                psf_grid=None, tol=1.e-3):
    """
    Update the flat field given current psf model
    """
    assert 0, 'Update: multithreading, dq, dataloss'
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
            flux, bkg_parms, bkg = fit_single_patch((data[i].ravel(),
                                                    psf.ravel(), flat.ravel(),
                                                    background))
            model = flat * (flux * psf + bkg.reshape(patch_shape))
            se = (data[i] - model) ** 2.
            norm[xpg + xpc[i], ypg + ypc[i]] += 1.
            new_flat[xpg + xpc[i], ypg + ypc[i]] += (data[i] / model) * \
                current_flat[xpg + xpc[i], ypg + ypc[i]]
            new_tse += se.sum()

        new_flat /= norm
        new_flat *= new_flat.size / new_flat.sum()

        if (tse - new_tse) / new_tse > tol:
            tse = new_tse
            current_flat = new_flat
        else:
            return new_flat, norm, new_tse

def fit_single_patch((data, psf, flat, ind, background)):
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

def data_loss(data, model, kind, floor, gain, var=None):
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

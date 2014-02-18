import multiprocessing
import numpy as np

from .grid_definitions import get_grids
from .generation import render_psfs
from .patch_fitting import fit_single_patch, data_loss
from scipy.optimize import fmin, fmin_powell


def update_psf(data, dq, current_flat, current_psf, 
               patch_centers, shifts, background, eps, threads, loss_kind,
               floor, gain, clip_parms):
    """
    Update the psf model, using bfgs.
    """
    global count
    count = 0

    psf_grid, patch_grid, pcs = get_grids(data.shape, current_flat.shape[0],
                                          data[0].shape)
    if patch_centers is None:
        patch_centers = pcs

    p0 = np.log(current_psf.ravel().copy())

    # powell is most efficient w/o gradients, just do one iteration
    res = fmin_powell(psf_loss, p0, maxiter=1,
                      args=(data, dq, current_flat, psf_grid, patch_grid,
                            patch_centers, shifts, background, eps, threads,
                            loss_kind, floor, gain, clip_parms))

    # get ssqe vector
    ssqe = psf_loss(res, data, dq, current_flat, psf_grid, patch_grid,
                    patch_centers, shifts, background, eps, threads, loss_kind,
                    floor, gain, clip_parms, summation=False)

    res = np.exp(res.reshape(current_psf.shape[0], current_psf.shape[1]))
    return res, ssqe

def psf_loss(psf_model, data, dq, current_flat, psf_grid, patch_grid,
             patch_centers, shifts, background, eps, threads, loss_kind, 
             floor, gain, clip_parms, summation=True):
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
            datum = data[i].ravel()
            psf = rendered_psfs[i].ravel()
            flat = current_flat[xpg + xpc[i], ypg + ypc[i]].ravel()
            flux, bkg_parms, bkg, ind = fit_single_patch((datum,
                                                          psf, flat,
                                                          dq[i].ravel(),
                                                          background, floor,
                                                          gain, clip_parms))
            model = flat * (flux * psf + bkg)
            s = data_loss(datum[ind], model[ind], loss_kind, floor, gain)
            ssqe[i] = np.sum(s)
    else:
        pool = multiprocessing.Pool(threads)
        mapfn = pool.map
        argslist = [None] * data.shape[0]
        for i in range(data.shape[0]):
            datum = data[i].ravel()
            psf = rendered_psfs[i].ravel()
            flat = current_flat[xpg + xpc[i], ypg + ypc[i]].ravel()
            argslist[i] = [datum, psf, flat,
                           dq[i].ravel(), background, floor, gain, clip_parms]

        results = list(mapfn(fit_single_patch, [args for args in argslist]))
        for i in range(data.shape[0]):
            datum = data[i].ravel()
            psf = rendered_psfs[i].ravel()
            flat = current_flat[xpg + xpc[i], ypg + ypc[i]].ravel()
            flux = results[i][0]
            bkg_parms = results[i][1]
            bkg = results[i][2]
            ind = results[i][3]
            model = flat * (flux * psf + bkg)
            ssqe[i] = np.sum(data_loss(datum[ind], model[ind], loss_kind,
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

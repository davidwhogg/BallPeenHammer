import multiprocessing
import numpy as np
import pyfits as pf

from .psf_update import update_psf
from .flat_update import update_flat
from .shifts_update import update_shifts

def PatchFitter(data, dq, ini_psf, ini_flat,
                patch_centers=None, background='linear', rendered_psfs=None,
                sequence=['shifts', 'flat', 'psf'], tol=1.e-4, eps=1.e-4,
                ini_shifts=None, shift_threads=1, psf_threads=1, floor=None,
                gain=None, maxiter=np.Inf, dumpfilebase=None, trim_frac=0.005,
                min_frac=0.75, loss_kind='nll-model', core_size=5,
                clip_parms=None, final_clip=[1, 4.]):
    """
    Patch fitting routines for BallPeenHammer.
    """
    assert background in [None, 'constant', 'linear']
    assert np.mod(data[0].shape[0], 2) == 1, 'Patch shape[0] must be odd'
    assert np.mod(data[0].shape[1], 2) == 1, 'Patch shape[1] must be odd'
    assert np.mod(core_size, 2) == 1, 'Core size must be odd'
    for i in range(len(sequence)):
        assert sequence[0] in ['shifts', 'flat', 'psf']

    # core inidices
    xcenter = (data[0].shape[0] - 1) / 2
    ycenter = (data[0].shape[1] - 1) / 2
    buff = (core_size - 1) / 2
    xcore = xcenter - buff, xcenter + buff + 1
    ycore = ycenter - buff, ycenter + buff + 1

    # initialize
    current_psf = ini_psf.copy()
    current_loss = np.Inf
    current_flat = ini_flat.copy()
    if ini_shifts is not None:
        ref_shifts = ini_shifts.copy()
    else:
        ref_shifts = np.zeros(data.shape[0])

    # minimum number of patches, mask initialization
    Nmin = np.ceil(min_frac * data.shape[0]).astype(np.int)
    mask = np.arange(data.shape[0]).astype(np.int)

    # run
    tot_ssqe = np.Inf
    iterations = 0
    while True:
        if clip_parms is None:
            cp = None
        else:
            try:
                cp = clip_parms[interations]
            except:
                cp = final_clip

        if iterations > maxiter:
            return current_flat, current_psf, shifts

        for kind in sequence:
            if kind == 'shifts':
                shifts, ssqe = update_shifts(data[:, xcore[0]:xcore[1],
                                                  ycore[0]:ycore[1]],
                                             dq[:, xcore[0]:xcore[1],
                                                ycore[0]:ycore[1]],
                                             current_flat, current_psf,
                                             data[0].shape,
                                             patch_centers, ref_shifts,
                                             background, shift_threads,
                                             loss_kind, floor, gain, None)
                if iterations == 0:
                    ref_shifts = shifts.copy()

                print 'Shift step done, pre ssqe: ', ssqe.sum()
                if (trim_frac is not None) & (mask.size > Nmin):
                    assert trim_frac > 0., 'trim_frac must be positive or None'
                    Ntrim = np.ceil(mask.size * trim_frac).astype(np.int)
                    if (mask.size - Ntrim < Nmin):
                        Ntrim = mask.size - Nmin

                    # sort and trim the arrays
                    ind = np.argsort(ssqe)[:-Ntrim]
                    mask = mask[ind]
                    data = data[ind]
                    dq = dq[ind]
                    ref_shifts = ref_shifts[ind]
                    if patch_centers is not None:
                        patch_centers = patch_centers[ind]

                    # re-run shifts
                    shifts, ssqe = update_shifts(data[:, xcore[0]:xcore[1],
                                                      ycore[0]:ycore[1]],
                                                 dq[:, xcore[0]:xcore[1],
                                                    ycore[0]:ycore[1]],
                                                 current_flat, current_psf,
                                                 data[0].shape,
                                                 patch_centers,
                                                 ref_shifts,
                                                 background, shift_threads,
                                                 loss_kind, floor, gain,
                                                 None)
                else:
                    ind = np.arange(data.shape[0])

                print 'Shift step done, post ssqe: ', ssqe.sum()

                if dumpfilebase is not None:
                    np.savetxt(dumpfilebase + '_mask_%d.dat' % iterations, ind,
                               fmt='%d')
                    np.savetxt(dumpfilebase + '_shifts_%d.dat' % iterations,
                               shifts)
                    np.savetxt(dumpfilebase + '_shift_ssqe_%d.dat' % iterations,
                               ssqe)

            if kind == 'psf':
                current_psf, ssqe = update_psf(data, dq,
                                               current_flat, current_psf,
                                               data[0].shape,
                                               patch_centers,
                                               shifts,
                                               background, eps, psf_threads,
                                               loss_kind, floor, gain,
                                               cp)

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

import multiprocessing
import numpy as np
import pyfits as pf

from .grid_definitions import get_grids
from .shifts_update import update_shifts
from .patch_fitting import evaluate
from .psf_update import update_psf
from .plotting import linesearchplot

def PatchFitter(data, dq, ini_psf, patch_shape, id_start, background='linear',
                sequence=['shifts', 'psf'], tol=1.e-4, eps=1.e-4,
                ini_shifts=None, Nthreads=16, floor=None, plotfilebase=None,
                gain=None, maxiter=np.Inf, dumpfilebase=None, trim_frac=0.005,
                min_frac=0.75, loss_kind='nll-model', core_size=5, plot=False,
                clip_parms=None, final_clip=[1, 4.], q=0.5, clip_shifts=False,
                h=1.4901161193847656e-08, Nplot=20,
                shift_test_thresh=0.475):
    """
    Patch fitting routines for BallPeenHammer.
    """
    assert background in [None, 'constant', 'linear']
    assert np.mod(patch_shape[0], 2) == 1, 'Patch shape[0] must be odd'
    assert np.mod(patch_shape[1], 2) == 1, 'Patch shape[1] must be odd'
    assert np.mod(core_size, 2) == 1, 'Core size must be odd'
    assert (patch_shape[0] * patch_shape[1]) == data.shape[1], \
        'Patch shape does not match data shape'
    for i in range(len(sequence)):
        assert sequence[i] in ['shifts', 'psf', 'evaluate', 'plot_data']

    # set parameters
    parms = InferenceParms(h, q, eps, tol, gain, plot, floor, data.shape[0],
                           id_start, Nthreads, core_size, loss_kind, Nplot,
                           background, None, patch_shape, plotfilebase,
                           ini_psf.shape, shift_test_thresh)

    # initialize
    current_psf = ini_psf.copy()
    current_psf /= current_psf.max()
    current_loss = np.Inf
    if ini_shifts is not None:
        shifts = ini_shifts
        ref_shifts = ini_shifts.copy()
    else:
        ref_shifts = np.zeros(data.shape[0], 2)

    # minimum number of patches, mask initialization
    Nmin = np.ceil(min_frac * data.shape[0]).astype(np.int)
    mask = np.arange(data.shape[0], dtype=np.int)

    # run
    cost = np.Inf
    tot_cost = np.Inf
    iterations = 0
    while True:

        if iterations > maxiter:
            return current_psf, shifts

        for kind in sequence:
            if kind == 'shifts':
                if clip_shifts:
                    if clip_parms is None:
                        parms.clip_parms = None
                    else:
                        try:
                            parms.clip_parms = clip_parms[iterations]
                        except:
                            parms.clip_parms = final_clip
                    
                shifts, ssqe = update_shifts(data[:, parms.core_ind],
                                             dq[:, parms.core_ind],
                                             current_psf, ref_shifts, parms)

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
                    parms.Ndata = data.shape[0]

                    # re-run shifts
                    shifts, ssqe = update_shifts(data[:, parms.core_ind],
                                                 dq[:, parms.core_ind],
                                                 current_psf, ref_shifts, parms)
                else:
                    ind = np.arange(data.shape[0])

                print 'Shift step done, post ssqe: ', ssqe.sum()

                if dumpfilebase is not None:
                    np.savetxt(dumpfilebase + '_mask_%d.dat' % iterations, mask,
                               fmt='%d')
                    np.savetxt(dumpfilebase + '_shifts_%d.dat' % iterations,
                               shifts)
                    np.savetxt(dumpfilebase + '_shift_ssqe_%d.dat' % iterations,
                               ssqe)

            if kind == 'psf':
                if clip_parms is None:
                    parms.clip_parms = None
                else:
                    try:
                        parms.clip_parms = clip_parms[iterations]
                    except:
                        parms.clip_parms = final_clip

                current_psf, cost, line_ssqes, line_regs, line_scales = \
                    update_psf(data, dq, current_psf, shifts, parms)

                print 'Psf step done, ssqe: ', line_ssqes.min()

                if plot:
                    linesearchplot(line_ssqes, line_regs, line_scales,
                                   plotfilebase, iterations)

                if dumpfilebase is not None:
                    hdu = pf.PrimaryHDU(current_psf)
                    hdu.writeto(dumpfilebase + '_psf_%d.fits' % iterations,
                              clobber=True)

            if kind == 'evaluate':
                ssqe = evaluate((data, dq, shifts, current_psf, parms, False))
                print 'evaluated:', ssqe.sum()

            if kind == 'plot_data':
                if clip_parms is None:
                    parms.clip_parms = [1, np.inf]
                else:
                    try:
                        parms.clip_parms = clip_parms[iterations]
                    except:
                        parms.clip_parms = final_clip
                parms.plot = True
                ssqe = evaluate((data[:parms.Nplot], dq[:parms.Nplot],
                                 shifts[:parms.Nplot], current_psf, parms,
                                 False))
                parms.plot = False


        iterations += 1
        if iterations >= maxiter:
            return current_psf, shifts
"""
        if ((tot_cost - np.sum(cost)) / np.sum(cost)) < tol:
            return current_psf, shifts
        else:
            total_cost = np.sum(cost)
"""

class InferenceParms(object):
    """
    Class for storing and referencing parameters used in PSF inference.
    """
    def __init__(self, h, q, eps, tol, gain, plot, floor, Ndata, id_start, 
                 Nthreads, core_size, loss_kind, Nplot, background, 
                 clip_parms, patch_shape, plotfilebase, psf_model_shape,
                 shift_test_thresh):
        self.h = h
        self.q = q
        self.eps = eps
        self.tol = tol
        self.gain = gain
        self.plot = plot
        self.floor = floor
        self.Ndata = Ndata
        self.Nplot = Nplot
        self.Nthreads = Nthreads
        self.core_size = core_size
        self.loss_kind = loss_kind
        self.background = background
        self.clip_parms = clip_parms
        self.patch_shape = patch_shape
        self.plotfilebase = plotfilebase
        self.psf_model_shape = psf_model_shape
        self.shift_test_thresh = shift_test_thresh

        self.plot = False
        self.data_ids = range(id_start, Ndata + id_start)

        self.set_grids(core_size, patch_shape, psf_model_shape)

    def set_grids(self, core_size, patch_shape, psf_model_shape):
        """
        Set grid definitions for PSF and patches
        """
        # core foo
        ravel_size = patch_shape[0] * patch_shape[1]
        self.core_shape = (core_size, core_size)
        xcenter = (patch_shape[0] - 1) / 2
        ycenter = (patch_shape[1] - 1) / 2
        buff = (core_size - 1) / 2
        xcore = xcenter - buff, xcenter + buff + 1
        ycore = ycenter - buff, ycenter + buff + 1
        core_ind = np.arange(ravel_size, dtype=np.int).reshape(patch_shape)
        self.core_ind = core_ind[xcore[0]:xcore[1], ycore[0]:ycore[1]].ravel()

        # grid defs
        self.psf_grid, x = get_grids(patch_shape, psf_model_shape)


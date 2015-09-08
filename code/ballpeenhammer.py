import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from utils import FlatMapper

class BallPeenHammer(object):
    """
    Under a fixed PSF model, infer the underlying flat field.
    """
    def __init__(self, detector_shape, data, psfs, masks, patch_locs,
                 floor=0.05, gain=0.01, initial_flat=None, alpha_flat=None,
                 alpha_model=None, model_noise=False, est_noise=False):
        self.N = data.shape[0]
        self.D = data.shape[1]
        self.data = data
        self.gain = gain
        self.floor = floor
        self.patch_D = np.int(np.sqrt(data.shape[1]))
        self.flat_D = detector_shape[0] * detector_shape[1]
        self.detect_D = detector_shape
        self.masks = masks
        self.psfs = psfs
        self.patch_locs = patch_locs
        self.flat_model = initial_flat
        self.alpha_flat = alpha_flat
        self.alpha_model = alpha_model
        self.model_noise = model_noise
        if model_noise:
            self.est_noise = False
        else:
            self.est_noise = est_noise

        self.initialize(initial_flat)

        # ONLY SQUARE PATCHES FOR NOW!!!
        patch_side = np.int(np.sqrt(data[0].size))
        assert (patch_side - np.sqrt(data[0].size)) == 0., 'Non-square patches'
        self.flatmap = FlatMapper(self.flat_model, patch_side, patch_locs)

    def initialize(self, initial_flat):
        """
        Initialize the model.
        """
        # initial fluxes are the sum of the patch, backgrounds are zero.
        self.bkgs = np.zeros(self.N)
        self.fluxes = np.sum(self.data, axis=1)
        if initial_flat is None:
            self.flat_model = np.ones(self.detect_D)

    def optimize(self, model_noise=False, alpha=None, disp=0):
        """
        Use l_bfgs_b to minimize the log posterior.
        """
        from time import time
        if model_noise:
            raise NotImplementedError
        parms0 = np.append(np.append(self.flat_model.ravel(), self.fluxes),
                           self.bkgs)
        ini_nlp = self.neg_log_post(parms0, return_grads=False)
        t0 = time()
        result = fmin_l_bfgs_b(self.neg_log_post, parms0, disp=disp)
        self.flat_model = result[0][:self.flat_D]
        self.fluxes = result[0][self.flat_D:(self.flat_D + self.N)]
        self.bkgs = result[0][(self.flat_D + self.N):]
        print 'Optimized in %0.1f sec' % (time() - t0)
        print 'Initial Log Post. Prob.  : %0.2e' % ini_nlp
        print 'Optimized Log Post. Prob.: %0.2e' % result[1]
        return result

    def define_parms(self, args=None):
        """
        Create a single array of parameters suitable for fmin_l_bfgs_b.
        """
        # right now just defaulting to combining all.
        if args is None:
            return np.append(np.append(self.flat_model, self.fluxes),
                             self.bkgs)

    def neg_log_likelihood(self, flat, flxs, bkgs, return_grads=True):
        """
        Return the neg log likelihood of the data.  `parms` is an array with
        the flat, flux, and background parameters appended.
        """
        if self.model_noise:
            floor, gain = parms[-2:]
        else:
            floor, gain = self.floor, self.gain

        mapper = FlatMapper(flat, self.patch_D, self.patch_locs)
        flat_patches = mapper.get_1D_flat_patches()
        
        # model for the flux, pre-flat
        incident_model = flxs[:, None] * self.psfs + bkgs[:, None]
        
        models = flat_patches * incident_model

        if self.est_noise:
            x = self.data
        else:
            x = models
        
        C = floor + gain * x
        dmm = self.data - models
        dmm2 = dmm ** 2.

        nll = 0.5 * np.sum(dmm2 / C + np.log(C))
        if return_grads:
            m_grad = self.grad_model(models, dmm, dmm2, 1. / C, gain)
            flat_grad = self.grad_flat_nll(incident_model, m_grad,
                                           mapper.rowinds, mapper.colinds)
            flux_grad = self.grad_fluxes(flat_patches, self.psfs, m_grad)
            bkg_grad = self.grad_bkgs(flat_patches, m_grad)
            return nll, flat_grad, flux_grad, bkg_grad
        else:
            return nll, None, None, None
    
    def grad_model(self, models, dmm, dmm2, iC, gain):
        """
        Compute the gradient of the model wrt the nll.
        """
        grad = -2. * dmm * iC - gain * (dmm2 * iC ** 2. - iC)
        return 0.5 * grad

    def grad_flat_nll(self, incident_model, grad_model, row_ind, col_ind):
        """
        Compute the gradient of the flat wrt the nll.
        """
        grad = np.zeros(self.detect_D)
        for i in range(self.N):
            # it might be better to do this as one reshape.
            g = (incident_model[i] * grad_model[i]).reshape(self.patch_D,
                                                            self.patch_D)
            grad[row_ind[i], col_ind[i]] += g
        return grad.ravel()
        
    def grad_fluxes(self, flat_patches, psfs, grad_model):
        """
        Return the derivative of the fluxes wrt the nll.
        """
        return np.sum(flat_patches * psfs * grad_model, axis=1)

    def grad_bkgs(self, flat_patches, grad_model):
        """
        Return the derivative of the backgrounds wrt the nll.
        """
        return np.sum(flat_patches * grad_model, axis=1)

    def grad_check(self, Ncheck, parms, h=1.e-5):
        """
        Check the derivatives numerically.
        """
        perm = np.random.permutation
        nlp, ga = self.neg_log_post(parms)
        ind = np.append(np.append(perm(self.flat_D)[:Ncheck],
                                  (perm(self.N) + self.flat_D)[:Ncheck]),
                        (perm(self.N) + self.N + self.flat_D)[:Ncheck])
        for i in ind:
            if i < self.flat_D:
                ms = 'Flat grad'
            elif i > self.flat_D + self.N:
                ms = 'Bkg  grad'
            else:
                ms = 'Flux grad'
                
            # numerical grads
            p = parms.copy()
            p[i] += h
            nlp_f = self.neg_log_post(p, False)
            p[i] -= 2. * h
            nlp_b = self.neg_log_post(p, False)
            gn = (nlp_f - nlp_b) / 2. / h
            
            # assess
            check = np.abs(ga[i] - gn) / (np.abs(ga[i]) + np.abs(gn))
            check = check < 1.e-5 # magic

            print '%s, %5d: Anl %0.2e Num %0.2e %s' % (ms, i, ga[i], gn, check)

    def unpack_parms(self, parms):
        """
        Unpack the array of parameters to the respective flat, flux, and
        background components.
        """
        fluxes = parms[-2 * self.N:-1 * self.N]
        bkgs = parms[-1 * self.N:]
        if parms.size > 2 * self.N:
            flat = parms[:self.flat_D]
            modeling_flat = True
        else:
            flat = np.ones(self.flat_D)
            modeling_flat = False
        return flat, fluxes, bkgs

    def neg_log_post(self, parms, return_grads=True):
        """
        Return the negative log posterior probability
        """
        flat, fluxes, bkgs, modeling_flat = self.unpack_parms(parms)
        nlp, flat_g, flux_g, bkg_g = self.neg_log_likelihood(flat, fluxes,
                                                             bkgs)

        # prior on flat average to one
        # this just seems to work really poorly...
        if self.alpha_flat is not None:
            nlp += self.alpha_flat * (np.mean(flat) - 1) ** 2.
            if return_grads:
                flat_g = flat_g.reshape(self.detect_D)
                flat_g += 2. * self.alpha_flat * (flat - 1.)
                flat_g = flat_g.ravel()

        # prior that the flux and bkg should sum close
        # to the sum of the data.  Just trying to break degeneracies!
        if self.alpha_model is not None:
            totals = fluxes + bkgs
            diff = totals - np.sum(self.data, axis=1)
            nlp += self.alpha_model * np.sum(diff ** 2.)
            if return_grads:
                flux_g += 2. * self.alpha_model * diff
                bkg_g += 2. * self.alpha_model * diff

        # return the appropriate quantities.
        if return_grads:
            if modeling_flat:
                return nlp, np.append(np.append(flat_g, flux_g), bkg_g)
            else:
                return nlp, np.append(flux_g, bkg_g)
        else:
            return nlp

    def median_model(self):
        """
        Fit patches using least squares, median fractional residuals = flat
        """
        data = self.data
        psfs = self.psfs
        patch_locs = self.patch_locs
        noise_parms = self.noise_parms

        flats = np.zeros((psfs.shape[0], self.flat_model.shape[0], 
                          self.flat_model.shape[1]))

        for i in range(data.shape[0]):
            # fit and construct model
            fit_parms, _ = patch_fitter(data[i], psfs[i], None,
                                     noise_parms=noise_parms)
            model = psfs[i] * fit_parms[0] + fit_parms[1]

            # flat
            flat = data[i] / model
            ix, iy = self.flatmap.rowinds[i], self.flatmap.colinds[i]
            flats[i][ix, iy] = flat.reshape(patch_side, patch_side)

        # Gather result
        Nx = self.flat_model.shape[0]
        for i in range(Nx):
            fs = flats[:, i]
            y = np.apply_along_axis(lambda v: np.median(v[np.nonzero(v)]), 0,
                                    fs)
            self.flat_model[i] = y

def patch_fitter(data, psf, mask, flat=None, noise_parms=None, tol=1e-3):
    """
    Fit patch with a constant background and PSF.
    THIS DOES NOT MINIMIZE THE CORRECT OBJECTIVE, ONLY APPROX.
    """
    if noise_parms == None:
        ivar = np.ones_like(data) 
    else:
        ivar = 1. / (noise_parms[0] + noise_parms[1] * (psf * np.sum(data)))
    if flat == None:
        flat = np.ones_like(data)

    A = np.ones((psf.size, 2))
    A[:, 0] = psf * flat
    A[:, 1] = flat
    ATICA = np.dot(A.T, data * ivar)
    ATICY = np.linalg.inv(np.dot(A.T, A * ivar[:, None]))
    fit_parms = np.dot(ATICA, ATICY)
    if noise_parms == None:
        return fit_parms, ATICA

    model = np.dot(A, fit_parms)
    chi2_old = np.sum((data - model) ** 2. * ivar)
    dlt_chi2 = np.inf
    while dlt_chi2 > 0:
        ivar = 1. / (noise_parms[0] + noise_parms[1] * model)
        ATICA = np.dot(A.T, data * ivar)
        ATICY = np.linalg.inv(np.dot(A.T, A * ivar[:, None]))
        fit_parms = np.dot(ATICA, ATICY)
        model = np.dot(A, fit_parms)
        chi2_cur = np.sum((data - model) ** 2. * ivar)
        dlt_chi2 = chi2_old - chi2_cur
        chi2_old = chi2_cur
    return fit_parms, ATICA

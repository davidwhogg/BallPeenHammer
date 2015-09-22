import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from utils import FlatMapper
from time import time

class BallPeenHammer(object):
    """
    Under a fixed PSF model, infer the underlying flat field.
    """
    def __init__(self, detector_shape, data, psfs, masks, patch_locs,
                 floor=0.05, gain=0.01, initial_flat=None, alpha_flat=None,
                 alpha_model=None, alpha_delta=None, alpha_bkgs=None,
                 model_noise=False, est_noise=False):
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
        self.alpha_bkgs = alpha_bkgs
        self.alpha_flat = alpha_flat
        self.alpha_delta = alpha_delta
        self.alpha_model = alpha_model
        self.model_noise = model_noise
        if model_noise:
            self.est_noise = False
        else:
            self.est_noise = est_noise

        if self.masks == None:
            self.sum = np.sum
            self.dot = np.dot
        else:
            self.sum = self.quick_masked_sum
            self.dot = self.quick_masked_dot

        self.initialize(initial_flat)

        # ONLY SQUARE PATCHES FOR NOW!!!
        patch_side = np.int(np.sqrt(data[0].size))
        assert (patch_side - np.sqrt(data[0].size)) == 0., 'Non-square patches'
        self.flatmap = FlatMapper(self.flat_model, patch_side, patch_locs)

    def quick_masked_sum(self, arr, axis=None):
        """
        Use numpy's masked array functionality to do the sum.
        """
        ma = np.ma.MaskedArray(arr, mask=self.masks)
        return ma.sum(axis=axis)

    def quick_masked_dot(self, arr1, arr2):
        """
        Use numpy's masked array functionality to do the sum.
        """
        ma1 = np.ma.MaskedArray(arr1, mask=self.masks)
        ma2 = np.ma.MaskedArray(arr2, mask=self.masks)
        return np.ma.dot(ma1, ma2)

    def initialize(self, initial_flat):
        """
        Initialize the model.
        """
        # initial fluxes are the sum of the patch, backgrounds are zero.
        self.bkgs = np.zeros(self.N)
        self.fluxes = np.sum(self.data, axis=1)
        if initial_flat is None:
            self.flat_model = np.ones(self.detect_D)
        else:
            self.flat_model = initial_flat

    def optimize(self, model_flat=True, model_noise=False, alpha=None, disp=0):
        """
        Use l_bfgs_b to minimize the log posterior.
        """
        if model_noise:
            raise NotImplementedError
        parms0 = self.define_parms(model_flat)
        ini_nlp = self.neg_log_post(parms0, return_grads=False)
        t0 = time()
        result = fmin_l_bfgs_b(self.neg_log_post, parms0, disp=disp)
        delta_flat, fluxes, bkgs, _ = self.unpack_parms(result[0])
        self.flat_model = delta_flat + 1.
        self.fluxes = fluxes
        self.bkgs = bkgs
        print 'Optimized in %0.1f sec' % (time() - t0)
        print 'Initial Log Post. Prob.  : %0.2e' % ini_nlp
        print 'Optimized Log Post. Prob.: %0.2e' % result[1]
        return result

    def define_parms(self, model_flat=True):
        """
        Create a single array of parameters suitable for fmin_l_bfgs_b.
        """
        # right now just defaulting to combining all.
        if model_flat:
            return np.append(np.append(self.flat_model, self.fluxes),
                             self.bkgs)
        else:
            return np.append(self.fluxes, self.bkgs)

    def hdf5_save_parms(self, filename):
        """
        Save the current parameters to file.
        """
        import h5py
        f = h5py.File(filename, 'w')
        f.create_dataset('flat', data=self.flat_model)
        f.create_dataset('bkgs', data=self.bkgs)
        f.create_dataset('fluxes', data=self.fluxes)
        f.close()

    def hdf5_load_parms(self, filename):
        """
        Save the current parameters to file.
        """
        import h5py
        f = h5py.File(filename, 'r')
        self.flat_model = f['flat'][:]
        self.bkgs = f['bkgs'][:]
        self.fluxes = f['fluxes'][:]
        f.close()

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

        nll = 0.5 * self.sum(dmm2 / C + np.log(C))
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
        g_vals = incident_model * grad_model
        if self.masks is not None:
            g_vals[self.masks] = 0.

        for i in range(self.N):
            grad[row_ind[i], col_ind[i]] += g_vals[i].reshape(self.patch_D,
                                                              self.patch_D)

        return grad.ravel()
        
    def grad_fluxes(self, flat_patches, psfs, grad_model):
        """
        Return the derivative of the fluxes wrt the nll.
        """
        return self.sum(flat_patches * psfs * grad_model, axis=1)

    def grad_bkgs(self, flat_patches, grad_model):
        """
        Return the derivative of the backgrounds wrt the nll.
        """
        return self.sum(flat_patches * grad_model, axis=1)

    def grad_check(self, Ncheck, parms, h=1.e-6):
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
            delta_flat = parms[:self.flat_D].reshape(self.detect_D)
            modeling_flat = True
        else:
            delta_flat = self.flat_model - 1.
            modeling_flat = False
        return delta_flat, fluxes, bkgs, modeling_flat

    def neg_log_post(self, parms, return_grads=True):
        """
        Return the negative log posterior probability
        """
        delta_flat, fluxes, bkgs, modeling_flat = self.unpack_parms(parms)
        flat = 1. + delta_flat
        nlp, flat_g, flux_g, bkg_g = self.neg_log_likelihood(flat, fluxes,
                                                             bkgs)

        if self.alpha_bkgs is not None:
            prior_ivars = self.alpha_bkgs * np.sum(self.data, axis=1)
            nlp += np.sum(prior_ivars * bkgs ** 2.)
            if return_grads:
                bkg_g += 2. * prior_ivars * bkgs

        # prior on shifts...
        if self.alpha_delta is not None:
            nlp += self.alpha_delta * np.sum(delta_flat ** 2.)
            if return_grads:
                flat_g += 2. * self.alpha_delta * delta_flat.ravel()

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

    def blocked_gibbs(self, Nwalkers, Nsamps, ini_parms):
        """
        Do blocked gibbs sampling, alternating between flat and flux/bkg
        parameters.  Emcee is used for proposals, drawing samples.  
        """
        import emcee
        flat_pos = [ini_parms[:self.flat_D] *
                    (1. + 1.e-4 * np.random.randn(self.flat_D))
                    for _ in range(Nwalkers)]
        bright_pos = [ini_parms[self.flat_D:] *
                      (1. + 1.e-4 * np.random.randn(2 * self.N))
                      for _ in range(Nwalkers)]

        cur_sample = ini_parms
        print Nwalkers
        assert 0
        from time import time
        t0 = time()
        s = emcee.EnsembleSampler(Nwalkers, self.flat_D,
                                  self.log_post_given_brightnesses,
                                  args=[cur_sample[self.flat_D:]])
        s.run_mcmc(flat_pos, 1)
        print time()-t0

    def log_post_given_flat(self, brightness_parms, flat):
        """
        Compute the log of the posterior, conditioned on set flat parameters
        """
        parms = np.append(flat, brightness_parms)
        return -1. * self.neg_log_post(parms, False)

    def log_post_given_brightnesses(self, flat, brightness_parms):
        """
        Compute the log of the posterior, conditioned on set fluxes/bkgs.
        """
        parms = np.append(flat, brightness_parms)
        return -1. * self.neg_log_post(parms, False)

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

        def patch_fitter(self, flat_patches=None, noise_model=None):
            """
            Fit patchs with a constant background and PSF.
            """
            if noise_parms == None:
                ivars = np.ones_like(self.data) 
            else:
                ivars = 1. / noise_model
                
            if flat_patches == None:
                flat_patches = np.ones_like(self.data)

            fit_parms = np.zeros((self.N, 2))
            parm_uncs = np.zeros((self.N, 2))
                                 
            for i in range(self.N):
                A = np.ones((self.patch_D, 2))
                A[:, 0] = self.psfs[i] * flat_patches[i]
                A[:, 1] = flat_patches[i]
                ATICA = np.dot(A.T, self.data[i] * ivars[i])
                ATICY = np.linalg.inv(np.dot(A.T, A * ivars[i][:, None]))
                fit_parms[i] = np.dot(ATICA, ATICY)
                parm_uncs[i] = ATICA
            return fit_parms, parm_uncs

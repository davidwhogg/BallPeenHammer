

import sys
sys.path.append('/home/rfadely/flann/src/python')
sys.path.append('/home/rfadely/sdss-mixtures/code')
from pyflann import *
from patch import *

import time

import numpy as np
import pyfits as pf
import matplotlib.pyplot as pl

class Calibrator(object):
    """
    Add dflatclip comment
    shift_and_scale; shift and scale the NNs before using them. 
    """
    def __init__(self,filelist,outbase,
                 k=1,max_iters=1,
                 patchshape = (5,5),
                 trueflat=None,truedark=None,
                 truesky=None,
                 flat=None,dark=None,
                 hivar=True,lovar=True,
                 flatonly=False,darkonly=False,
                 shift_and_scale=True,
                 dflatclip=0.1,
                 Ngrid = 1,
                 nn_precision=0.99):

        assert patchshape[0] % 2 != 0, 'Patch xside must be odd'
        assert patchshape[1] % 2 != 0, 'Patch yside must be odd'

        
        self.k = k
        self.flat = flat
        self.dark = dark
        self.iters = 0
        self.lovar = lovar
        self.hivar = hivar
        self.Ngrid = Ngrid
        self.Nimages = len(filelist)
        self.outbase = outbase
        self.truesky = truesky
        self.trueflat = trueflat
        self.truedark = truedark
        self.flatonly = flatonly
        self.darkonly = darkonly
        self.filelist = filelist
        self.max_iters = max_iters
        self.dflatclip = dflatclip
        self.patchshape = patchshape
        self.nn_precision = nn_precision
        self.shift_and_scale = shift_and_scale
        self.img2patch_inds = {}

        # Save 'true' flat/dark
        h = pf.PrimaryHDU(self.trueflat)
        h.writeto(self.outbase+'_trueflat.fits')
        h = pf.PrimaryHDU(self.truedark)
        h.writeto(self.outbase+'_truedark.fits')

        self.run_calibrator()

    def run_calibrator(self):

        for i in range(self.max_iters):
            titer0 = time.time()
            self.load_images()
            if self.trueflat!=None:
                self.break_images()
            self.calibrate_images()
            self.patchify()
            self.sort()
            self.images = 0.0 # dump images
            t0 = time.time()
            print 'Starting NNs lookup'
            self.nns()
            print 'Done NNs, took %f' % (time.time()-t0)
            self.make_mask()
            self.calibration_step()

            self.iters += 1

            self.delta_dark = self.delta_dark.reshape(self.imgshape)
            self.delta_flat = self.delta_flat.reshape(self.imgshape)

            # change flat/dark models
            # BUG possble, multiplicative change to flat
            self.flat *= np.clip(self.delta_flat, 
                                 (1.-self.dflatclip),
                                 (1.+self.dflatclip))
            self.dark += self.delta_dark

            # No flat pixel can go negative
            assert np.all(self.flat>0)

            # renormalize
            pad0,pad1 = self.patchshape
            pad0 = (pad0-1)/2
            pad1 = (pad1-1)/2
            self.dark[pad0:-pad0,pad1:-pad1] = self.dark[pad0:-pad0,pad1:-pad1] - np.mean(self.dark[pad0:-pad0,pad1:-pad1])
            self.flat[pad0:-pad0,pad1:-pad1] = self.flat[pad0:-pad0,pad1:-pad1] / np.mean(self.flat[pad0:-pad0,pad1:-pad1])

            # write out calib image for step
            h = pf.PrimaryHDU(self.delta_dark)
            h.writeto(self.outbase+'_Ddark_'+str(self.iters)+'.fits')
            h = pf.PrimaryHDU(self.delta_flat)
            h.writeto(self.outbase+'_Dflat_'+str(self.iters)+'.fits')
            h = pf.PrimaryHDU(self.dark)
            h.writeto(self.outbase+'_dark_'+str(self.iters)+'.fits')
            h = pf.PrimaryHDU(self.flat)
            h.header.update('PATCH0',self.patchshape[0])
            h.header.update('PATCH1',self.patchshape[1])
            h.writeto(self.outbase+'_flat_'+str(self.iters)+'.fits')
            print 'Iter %f took %fs' % (self.iters,time.time()-titer0)

    def load_images(self):
        """
        Create numpy array of image data
        """

        for i in range(self.Nimages):
            f = pf.open(self.filelist[i])
            if i==0:
                self.images = np.empty((self.Nimages,
                                        f[0].data.shape[0],
                                        f[0].data.shape[1])) 
                self.imgshape = (f[0].data.shape[0],
                                 f[0].data.shape[1])

            self.images[i] = f[0].data
            f.close()
            if i%100==0: print 'Loading images',i

        self.Npix = self.images[0].shape[0] * \
            self.images[0].shape[1]

    def break_images(self):
        """
        Set 'truth'
        """
        if self.truedark==None:
            self.truedark = np.zeros(self.imgshape)
        if self.truesky==None:
            self.truesky = np.zeros(self.imgshape)

        for i in range(self.images.shape[0]):
            self.images[i] += self.truesky
            self.images[i] *= self.trueflat
            self.images[i] += self.truedark


    def calibrate_images(self):
        """
        Calibrate data with current flat, dark models
        """
        # simple initial flat, darks if not given
        if self.flat==None:
            print 'Initializing flat as ones'
            self.flat = np.ones(self.images[0].shape)
        if self.dark==None:
            print 'Initializing \'dark\' as zeros'
            self.dark = np.zeros(self.images[0].shape) 

        for i in range(self.images.shape[0]):
            self.images[i] -= self.dark
            self.images[i] /= self.flat

        if self.iters==0:
            # Save initial flat/dark
            h = pf.PrimaryHDU(self.flat)
            h.writeto(self.outbase+'_initflat.fits')
            h = pf.PrimaryHDU(self.truedark)
            h.writeto(self.outbase+'_initdark.fits')

    def patchify(self):
        """
        Make patches from images
        """
        for i in range(self.images.shape[0]):
            d,ids = self.patchify_one(i,self.images[i])
            if i==0:
                Npatches = d.shape[0]
                self.data = np.zeros((Npatches*self.Nimages,
                                      d.shape[1]))
                self.ids = np.zeros(Npatches*self.Nimages,dtype='int')

            self.ids[i*Npatches:(i+1)*Npatches] = ids
            self.data[i*Npatches:(i+1)*Npatches,:] = d
            if i%100==0: print 'Patchifying',i

        assert np.all(self.data[-1,:]!=0.0)

    def patchify_one(self,i,d):
        """
        Create patches from one image, cut on variance
        """
        patchshape = self.patchshape # kill me now
        cpx = (self.patchshape[0] * self.patchshape[1] - 1) / 2
        p = Patches(d,np.ones(d.shape),pshape=patchshape,flip=False)
    
        # record the indices of the variance cuts once
        if self.iters==0:
            var = np.var(p.data,axis=1)
            v   = np.sort(var)
            thresh = v[0.99 * v.shape[0]] # magic number 0.99 from 
                                          # inspecting var dist.
            ind = np.where(var>thresh)[0]
            if self.hivar:
                self.img2patch_inds[i] = ind
            if self.lovar:
                idx = np.random.shuffle(np.where(var<thresh)[0])
                idx = idx[:p.data[ind,:].shape[0]]
                if self.hivar:
                    self.img2patch_inds[i] = np.append(ind,idx)
                else:
                    self.img2patch_inds[i] = idx

        inds = self.img2patch_inds[i]
        return p.data[inds,:],p.indices[inds,cpx]


    def sort(self):
        """
        Sort patches in pixel ID order
        """
        ind = np.argsort(self.ids)
        self.data = self.data[ind]
        self.ids  = self.ids[ind]
        self.bins = np.bincount(self.ids)

        coverage = np.zeros(self.imgshape).ravel()
        coverage[:self.ids.max()+1] += self.bins
        h = pf.PrimaryHDU(coverage.reshape(self.imgshape))
        h.writeto(self.outbase+'_coverage_'+str(self.iters)+'.fits')

    def nns(self):
        """
        Use flann to calculate kNNs
        """
        # center pixel values
        cpx = (self.patchshape[0] * self.patchshape[1] - 1) / 2
        vals = self.data[:,cpx]
        # non-center (outer) pixel values
        opx = np.append(np.arange(cpx, dtype=int),
                        np.arange(cpx, dtype=int) + cpx + 1)
        data = self.data[:,opx]
        
        if (self.iters==0) & (self.Ngrid>1):
            minid = np.min(self.ids)
            maxid = np.max(self.ids)
            gridstep = (maxid-minid)/self.Ngrid**2
            self.nn_inds = np.zeros((self.data.shape[0],
                                    (self.Ngrid**2-1)*self.k)).astype(int)
            self.nn_dists = np.zeros((self.data.shape[0],
                                      (self.Ngrid**2-1)*self.k))
            # ugh... indices inside zones
            self.grid_inds = {}
            for i in range(self.Ngrid**2):
                lbound = minid+gridstep*i
                ubound = lbound + gridstep
                
                ininds = np.where((self.ids>=lbound) & 
                                  (self.ids<ubound))[0]

                self.grid_inds[i] = ininds
    
        flann = FLANN()
        for i in range(self.Ngrid**2):

            if self.Ngrid==1:
                # No grids, one shot
                parms = flann.build_index(data,
                                          target_precision=self.nn_precision,
                                          log_level='info')
                inds, dists = flann.nn_index(data,self.k+1,
                                             checks=parms['checks'])
                # slice out self matches
                self.nn_inds = inds[:,1:]
                self.nn_dists = dists[:,1:]
            else:
                # brutal
                idx = 0
                for j in range(self.Ngrid**2):
                    if i!=j:
                        tstind = self.grid_inds[i]
                        trnind = self.grid_inds[j]
            

                        # build lookup for neighbors not in grid zone
                        parms = flann.build_index(data[trnind,:],
                                                  target_precision=self.nn_precision,
                                                  log_level='info')

                        # evaluate for grid zone
                        inds, dists = flann.nn_index(data[tstind,:],self.k,
                                                     checks=parms['checks'])
                        if self.k==1:
                            inds = np.atleast_2d(inds).T
                            dists = np.atleast_2d(dists).T

                        self.nn_inds[tstind,idx*self.k:(idx+1)*self.k] = inds
                        self.nn_dists[tstind,idx*self.k:(idx+1)*self.k] = dists
                        idx += 1

        self.cpx_vals = vals
        self.cpx_nns  = vals[self.nn_inds]



        # untested spaghetti code from Hogg
        if self.shift_and_scale:
            self.patch_means = np.mean(data,axis=1)
            dd = data - self.patch_means[:,None]
            dn = data[self.nn_inds] - self.patch_means[self.nn_inds][:,:,None]
            scale = np.sum(dd[:,None,:]*dn,axis=2) / np.sum(dn*dn,axis=2)
            self.cpx_nns = (self.cpx_nns - self.patch_means[self.nn_inds]) * \
                scale + self.patch_means[:,None]

    def make_mask(self):
        """
        Masking around edges
        """
        mask = np.ones((np.sqrt(self.Npix),
                        np.sqrt(self.Npix)),
                       dtype='int')
        xpad = (self.patchshape[0]-1)/2
        ypad = (self.patchshape[1]-1)/2

        mask[:xpad,:] = 0
        mask[-xpad:,:] = 0
        mask[:,:ypad] = 0
        mask[:,-ypad:] = 0
        
        self.mask = mask.ravel()


    def calibration_step(self):
        """
        Calculate the delta dark, flat
        """
        if self.iters==0:
            rndpix = np.random.permutation(np.arange(self.ids.min(),
                                                     self.ids.max())
                                           .astype(int))[:10]
            self.rndpix = rndpix

        self.delta_dark = self.dark.ravel() * 0.0 
        self.delta_flat = self.flat.ravel() * 0.0 + 1.0

        curr = 0
        idx = np.arange(self.ids.max()+1)
        for i in range(self.bins.shape[0]):
            if (self.mask[idx[i]]>0) & (self.bins[i]>2):

                zerop, slope = 0., 1.

                nns = self.cpx_nns[curr: \
                                   curr+self.bins[i],:]

                # these are the inverse variance and mean of
                # the nns for each patch that hits pix
                iv = 1. / np.var(nns,axis=1)
                m = np.mean(nns,axis=1)
                # the data
                vals = self.cpx_vals[curr:curr+self.bins[i]]
                # weighted least squares
                if self.flatonly:
                    slope = np.dot(m * iv, vals) / np.dot(m * iv, m)            
                elif self.darkonly:
                    A = np.atleast_2d(np.ones_like(m))
                    ATAinv = np.linalg.inv(np.dot(A * iv, A.T))
                    ATb = np.dot(A * iv, vals)
                    rs = np.dot(ATAinv, ATb)
                    zerop = rs[0]
                else:
                    A = np.vstack((np.ones_like(m), m))
                    ATAinv = np.linalg.inv(np.dot(A * iv, A.T))
                    ATb = np.dot(A * iv, vals)
                    rs = np.dot(ATAinv, ATb)
                    zerop = rs[0]
                    slope = rs[1]
                self.delta_dark[idx[i]] = zerop
                # BUG possible, one minus?
                self.delta_flat[idx[i]] = slope
            if self.ids[curr] in self.rndpix:
                out = np.empty((m.shape[0],5))
                out[:,0] = m
                out[:,1] = iv
                out[:,2] = vals
                out[:,3] = np.zeros_like(m) + zerop
                out[:,4] = np.zeros_like(m) + slope
                np.savetxt(self.outbase+'_rndnninfo_'+str(self.ids[curr])
                           +'_'+str(self.iters)+'.txt',out)
                
                
            if i%(self.Npix/64)==0: 
                print 'Calibrated pixel %6.0f, %1.2f of total' % \
                    (i,float(curr)/self.data.shape[0])
            curr += self.bins[i]


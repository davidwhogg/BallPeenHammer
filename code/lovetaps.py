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

    def __init__(self,filelist,k=8,max_iters=1,
                 patchshape = (5,5),
                 flat=None,dark=None,
                 hivar=True,lovar=True,
                 flatonly=False,
                 nn_precision=0.99):

        assert patchshape[0] % 2 != 0, 'Patch side must be odd'
        assert patchshape[0]==patchshape[1], \
            'Patch must be square'


        self.k = k
        self.flat = flat
        self.dark = dark
        self.iters = 0
        self.lovar = lovar
        self.hivar = hivar
        self.flatonly = flatonly
        self.Nimages = len(filelist)
        self.filelist = filelist
        self.max_iters = max_iters
        self.patchshape = patchshape
        self.nn_precision = nn_precision

        self.run_calibrator()

    def run_calibrator(self):

        # below here will be moved to 
        # an iterated loop
        self.load_images()
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

        self.dd1 = self.delta_dark.copy()
        self.df1 = self.delta_flat.copy()

        # change flat/dark models
        self.flat += self.df1.reshape(self.imgshape)
        self.dark += self.dd1.reshape(self.imgshape)

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

        for i in range(self.images.shape[0]):
            self.images[i] += self.dark
            self.images[i] *= self.flat

        self.flat = None
        self.dark = None

    def calibrate_images(self):
        """
        Calibrate data with current flat, dark models
        """
        # simple initial flat, darks if not given
        if self.flat==None:
            self.flat = np.ones(self.images[0].shape)
        if self.dark==None:
            ind = np.random.permutation(len(self.images))
            ind = ind[:0.1 * ind.shape[0]] # use a tenth for 
                                           # dark estimation
            self.dark = np.zeros(self.images[0].shape) + \
                np.median(self.images[ind])

        for i in range(self.images.shape[0]):
            self.images[i] -= self.dark
            self.images[i] /= self.flat

    def patchify(self):
        """
        """
        for i in range(self.images.shape[0]):
            d,ids = self.patchify_one(self.images[i])
            if i==0:
                Npatches = d.shape[0]
                self.data = np.zeros((Npatches*self.Nimages,
                                      d.shape[1]))
                self.ids = np.zeros(Npatches*self.Nimages,dtype='int')

            self.ids[i*Npatches:(i+1)*Npatches] = ids
            self.data[i*Npatches:(i+1)*Npatches,:] = d
            if i%100==0: print 'Patchifying',i

        assert np.all(self.data[-1,:]!=0.0)

    def patchify_one(self,d):
        """
        Create patches from image
        """
        patchshape = self.patchshape # kill me now
        p = Patches(d,np.ones(d.shape),pshape=patchshape,flip=False)
    
        var = np.var(p.data,axis=1)
        v   = np.sort(var)
        thresh = v[0.99 * v.shape[0]] # magic number 0.99 from 
                                      # inspecting var dist.
        hi = np.array([])
        lo = np.array([])
        if self.hivar:
            # get high var patches
            ind = var>thresh
            hi  = p.data[ind,:]
            hids = p.indices[ind,(patchshape[0]*patchshape[1]-1)/2+1] # odd patch
        if self.lovar:
            # get low var patches
            ind = var<thresh
            lo  = p.data[ind,:]
            lids = p.indices[ind,(patchshape[0]*patchshape[1]-1)/2+1] # odd patch
            ind = np.random.permutation(lo.shape[0])
            lo  = lo[ind[:hi.shape[0]],:]
            lids = lids[ind[:hi.shape[0]]]

        if (self.hivar) & (self.lovar):
            outd = np.concatenate((hi,lo))
            outids = np.concatenate((hids,lids))
            return outd,outids
        if (self.hivar):
            return hi,hids
        if (self.lovar):
            return lo,lids

    def sort(self):

        ind = np.argsort(self.ids)
        self.data = self.data[ind]
        self.ids  = self.ids[ind]
        self.bins = np.bincount(self.ids)

    def nns(self):
        pix = (self.patchshape[0] ** 2 - 1) / 2
        vals = self.data[:,pix]
        foo = np.append(np.arange(pix, dtype=int),
                        np.arange(pix, dtype=int) + pix + 1)
        data = self.data[:,foo]

        flann = FLANN()
        parms = flann.build_index(data,target_precision=self.nn_precision,
                                  log_level='info')

        inds, dists = flann.nn_index(data,self.k,checks=parms['checks'])
        self.nn_inds  = inds[:,1:]
        self.nn_dists = dists[:,1:]
        self.cpx_vals = vals
        self.cpx_nns  = vals[self.nn_inds]

    def make_mask(self):

        mask = np.ones((np.sqrt(self.Npix),
                        np.sqrt(self.Npix)),
                       dtype='int')
        Npad = (self.patchshape[0]-1)/2
        for i in range(Npad):
            mask[i,:] = 0
            mask[:,i] = 0
            i += 1
            mask[-i,:] = 0
            mask[:,-i] = 0

        self.mask = mask.ravel()


    def calibration_step(self):


        self.delta_dark = self.dark.ravel() * 0.0 
        self.delta_flat = self.flat.ravel() * 0.0 

        ncal = 0
        curr = 0
        idx = np.arange(self.ids.max()+1)
        for i in range(self.bins.shape[0]):
            if (self.mask[idx[i]]>0) & (self.bins[i]>1):
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
                    self.delta_flat[idx[i]] = np.dot(m * iv, vals) / np.dot(m * iv, m) - 1.
                else:
                    A = np.vstack(np.ones_like(m), m)
                    ATAinv = np.linalg.inv(np.dot(A * iv, A.T))
                    ATb = np.dot(A * iv, vals)
                    rs = np.dot(ATAinv, ATb)
                    self.delta_dark[idx[i]] = rs[0]
                    self.delta_flat[idx[i]] = rs[1] - 1
                ncal += 1
            if i%(self.Npix/64)==0: print i,self.delta_dark.min(),self.delta_dark.max()
                #print 'Calibrated pixel %6.0f, %1.2f of total' % \
                #    (i,float(curr)/self.data.shape[0])
            curr += self.bins[i]

        print ncal,np.unique(self.ids).shape
        ind = self.bins!=0
        print np.median(self.bins[ind])
        print np.min(self.bins[ind])
        print np.max(self.bins[ind])
        print np.std(self.bins[ind])

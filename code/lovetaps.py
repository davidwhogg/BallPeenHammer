import sys
sys.path.append('/home/rfadely/flann/src/python')
sys.path.append('/home/rfadely/sdss-mixtures/code')
from pyflann import *
from patch import *

import time

import numpy as np
import pyfits as pf

class Calibrator(object):

    def __init__(self,filelist,k=8,iters=1,
                 patchshape = (5,5),
                 flat=None,dark=None,
                 nn_precision=0.99):

        assert patchshape[0] % 2 != 0, 'Patch side must be odd'
        assert patchshape[0]==patchshape[1], \
            'Patch must be square'


        self.k = k
        self.flat = flat
        self.dark = dark
        self.filelist = filelist
        self.patchshape = patchshape
        self.nn_precision = nn_precision

        self.load_images()
        self.calibrate_images()
        self.patchify()
        self.nn_model()

    def load_images(self):
        """
        Create numpy array of image data
        """
        self.images = []
        for i in range(len(self.filelist)):
            f = pf.open(self.filelist[i])
            self.images.append(f[0].data)
            f.close()
            if i%100==0: print 'Loading images',i

        self.images = np.array(self.images)

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
        Patch up images, this is slow and 
        takes about 1.6s per (500x500) field.
        """
        self.xs = []
        self.ys = []
        self.data = []
        for i in range(self.images.shape[0]):
            d,x,y = self.patchify_one(self.images[i],
                                      self.patchshape)
            self.xs.append(x)
            self.ys.append(y)
            self.data.append(d)
            if i%100==0: print 'Patchifying',i
            
        self.xs = np.concatenate(self.xs)
        self.ys = np.concatenate(self.ys)
        self.data = np.concatenate(self.data)

    def patchify_one(self,d,patchshape):
        """
        Create patches from image
        """
        p = Patches(d,np.ones(d.shape),pshape=patchshape,flip=True)
    
        var = np.var(p.data,axis=1)
        var = np.sort(var)
        thresh = var[0.99 * var.shape[0]] # magic number 0.99 from 
                                          # inspecting var dist.
            
        var = np.var(p.data,axis=1)
        # get high var patches
        ind = var>thresh
        hi  = p.data[ind,:]
        hxs = p.xs[ind].min(axis=1) + 2
        hys = p.ys[ind].min(axis=1) + 2
        # get low var patches
        ind = var<thresh
        lo  = p.data[ind,:]
        lxs = p.xs[ind].min(axis=1) + 2
        lys = p.ys[ind].min(axis=1) + 2
        ind = np.random.permutation(lo.shape[0])
        lo  = lo[ind[:hi.shape[0]],:]
        lxs = lxs[ind[:hi.shape[0]]]
        lys = lys[ind[:hi.shape[0]]]
        
        outd = np.concatenate((hi,lo))
        outx = np.concatenate((hxs,lxs))
        outy = np.concatenate((hys,lys))

        return outd,outx,outy

    def get_nn(self,test,train,k,target_precision):

        flann = FLANN()
        parms = flann.build_index(train,target_precision=target_precision,
                                  log_level='info')
        return flann.nn_index(test,k,checks=parms['checks'])

    def nn_model(self):
        """
        Painfully slow!
        """

        self.pred_means = []
        self.pred_vars  = []
        self.test_vals  = []
        
        x,y = np.meshgrid(range(self.images.shape[1]),
                                    range(self.images.shape[2]))
        x = x.flatten()
        y = y.flatten()

        
        pix = (self.patchshape[0] ** 2 - 1) / 2
        half = (self.patchshape[0]-1) / 2 + 1

        for i in range(self.images.shape[1]*
                       self.images.shape[2]):

            ind = (self.xs==x[i]) & (self.ys==y[i])
            if (self.data[ind].shape[0]>0):
                test = self.data[ind]
                vals = test[:,pix]
                test = np.delete(test,pix,axis=1)

                ind = ind == False
                train = self.data[ind]
                pred = train[:,pix]
                train = np.delete(train,pix,axis=1)
                ind, dist = self.get_nn(test,train,
                                        self.k,self.nn_precision)

                means = pred[ind].sum(axis=1)/self.k
                vars = pred[ind].var(axis=1)

            else:
                vals,means,vars = None, None, None

            self.test_vals.append(vals)
            self.pred_means.append(means)
            self.pred_vars.append(vars)

            if i%100==0: print 'NN ',i
                 

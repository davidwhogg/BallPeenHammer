import numpy as np

class FakeField(object):

    def __init__(self,flat,dark,
                 uninoise,
                 noisefact,
                 index = 2,
                 Nmean=3000,
                 fluxrange=(50,1000),
                 psfsigma=1.5,
                 window=4):

        self.psfsigma = psfsigma
        self.N = np.random.poisson(Nmean)
        self.N = np.int(np.round(self.N))
        self.fluxrange = fluxrange
        self.uninoise = uninoise
        self.noisefact = noisefact
        self.index = index
        self.window = window

        assert flat.shape == dark.shape, \
            'Flat and dark are not same shape'
        self.shape = flat.shape
        self.flat = flat
        self.dark = dark

        self.make_catalog()
        self.make_image()


    def add_gaussianpsfs(self):
        """
        Add gaussian psfs to image, this has been
        tried as a broadcast but takes a lot of memory and time
        """

        x, y = np.meshgrid(range(self.shape[0]),range(self.shape[1]))

        for i in range(self.N):
            ind = (x<=self.x0s[i] + self.window*self.psfsigma) & (x>=self.x0s[i] - self.window*self.psfsigma) & \
                (y<=self.y0s[i] + self.window*self.psfsigma) & (y>=self.y0s[i] - self.window*self.psfsigma)
            self.image[ind] += self.fluxes[i] * np.exp(-0.5 * ((x[ind]-self.x0s[i]) ** 2 +
                                                        (y[ind]-self.y0s[i]) ** 2)
                        / self.psfsigma ** 2) / np.sqrt(2. * np.pi * self.psfsigma ** 2)

    def draw_fluxes(self):
        r = np.random.rand(self.N)
        na = 1. - self.index
        L = self.fluxrange[0]
        H = self.fluxrange[1]
        self.fluxes = (L ** na + r * (H ** na - L ** na)) ** (1./na)

    def make_catalog(self):
        # draw N from Poisson with mean self.meanNumber
        # draw fluxes
        # draw x0s and y0s from uniforms.

        self.draw_fluxes()

        # this needs to go outside of image!!
        self.x0s = np.random.rand(self.N) * (self.shape[0] - 1)
        self.y0s = np.random.rand(self.N) * (self.shape[1] - 1)

    def make_image(self):
        
        self.shape = self.flat.shape
        self.image = np.zeros(self.shape)

        self.add_gaussianpsfs()

        self.true = self.image.copy()
        self.var = self.uninoise**2 + self.noisefact**2 * self.true
        self.image *= self.flat
        self.image += self.dark
        self.image += np.random.randn(self.shape[0],
                                      self.shape[1]) \
                                      * np.sqrt(self.var)

if __name__=='__main__':
    import matplotlib.pyplot as pl
    import pyfits as pf

    # some magic numbers here
    # gives ~25 S/N > 10 stars per
    # 10k pixels
    side = 500
    flat = np.ones(side**2).reshape((side,side))
    flat = np.linspace(0.90,1.1,side)
    flat = np.tile(flat,(side,1))
    dark = np.atleast_2d(np.linspace(-0.03,0.03,side))
    dark = np.tile(dark.T,(1,side))
    hwhm = 1.5
    f = FakeField(flat,dark,
                  0.03,
                  0.03,
                  psfsigma=hwhm,
                  Nmean=1650,
                  fluxrange=(0.02,8.0))

    q = np.sqrt(f.fluxes**2/f.uninoise**2/(2.35 * f.psfsigma)**2)
    ind = q > 10
    print q[ind]
    print q[ind].shape
    print f.fluxes.shape

    hdu = pf.PrimaryHDU(f.image)
    hdu.writeto('fooimage.fits')
    hdu = pf.PrimaryHDU(f.true)
    hdu.writeto('footrue.fits')

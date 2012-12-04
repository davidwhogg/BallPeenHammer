import numpy as np

class FakeField(object):

    def __init__(self,flat,dark,
                 uninoise,
                 noisefact,
                 index = 2,
                 Nrange=(300,1000),
                 fluxrange=(50,1000),
                 psfsigma=1.5):

        self.psfsigma = psfsigma
        self.N = np.random.rand() * (Nrange[1]-Nrange[0]) + \
            Nrange[0]
        self.N = np.int(np.round(self.N))
        self.fluxrange = fluxrange
        self.uninoise = uninoise
        self.noisefact = noisefact
        self.index = index

        assert flat.shape == dark.shape, \
            'Flat and dark are not same shape'
        self.flat = flat
        self.dark = dark

        self.make_image()


    def gaussianpsf(self,flux,xgrid,ygrid,x0,y0,psfsigma):
        """
        Return a gaussian psf on grid of pixels
        """
        psf = flux * np.exp(-0.5 * ((xgrid-x0) ** 2 + (ygrid-y0) ** 2)
                            / psfsigma ** 2) / np.sqrt(2. * np.pi * psfsigma ** 2)
        return psf

    def gaussianpsfs(self,xgrid,ygrid,x0,y0,psfsigma)

        # needs to go outside image
        self.x0s = np.random.rand(self.N) * (self.shape[0]-1),
        self.y0s = np.random.rand(self.N) * (self.shape[1]-1),
        y0 = np.random.rand() * (self.shape[1]-1),
            source = self.gaussianpsf(flux,x,y,x0,y0,self.psfsigma)
            self.image += source

        
        psf = self.fluxes[None,None,:] * np.exp(-0.5 * ((xgrid[:,:,None]-self.x0s[None,None,:]) ** 2 + (ygrid[:,:,None]-self.y0s[None,None,:]) ** 2)
                        / psfsigma ** 2) / np.sqrt(2. * np.pi * psfsigma ** 2)


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

    def make_image(self):
        
        self.shape = self.flat.shape
        self.image = np.zeros(self.shape)

        nx, ny = self.shape
        x, y = np.meshgrid(range(nx),range(ny))

        self.draw_fluxes()

        for i in range(self.N):
            flux = self.fluxes[i]
            x0 = np.random.rand() * (self.shape[0]-1),
            y0 = np.random.rand() * (self.shape[1]-1),
            source = self.gaussianpsf(flux,x,y,x0,y0,self.psfsigma)
            self.image += source

        self.true = self.image
        self.var = self.uninoise**2 + self.noisefact**2 * self.true
        self.image *= self.flat
        self.image += self.dark
        self.image += np.random.randn(self.shape[0],
                                      self.shape[1]) \
                                      * np.sqrt(self.var)

if __name__=='__main__':
    import matplotlib.pyplot as pl
    import pyfits as pf

    side = 500
    flat = np.ones(side**2).reshape((side,side))
    flat = np.linspace(0.90,1.1,side)
    flat = np.tile(flat,(side,1))
    dark = np.ones(side**2).reshape((side,side))
    hwhm = 1.5
    f = FakeField(flat,dark,
                  0.03,
                  0.03,
                  psfsigma=hwhm,
                  Nrange=(150,300),
                  fluxrange=(1,1.5))
    #pl.imshow(f.image,interpolation='nearest')
    #pl.gray()
    #pl.show()

    print f.fluxes**2/f.uninoise**2/(2.35 * f.psfsigma)**2
    print f.fluxes.shape

    hdu = pf.PrimaryHDU(f.image)
    hdu.writeto('foo.fits')

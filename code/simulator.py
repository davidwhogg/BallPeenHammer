import numpy as np

class FakeField(object):

    def __init__(self,flat,dark,
                 uninoise,
                 noisefact,
                 Nrange=(300,1000),
                 scorerange=(50,1000),
                 psfhwhm=1.5):

        self.psfhwhm = psfhwhm
        self.N = np.random.rand() * (Nrange[1]-Nrange[0]) + \
            Nrange[0]
        self.N = np.int(np.round(self.N))
        self.scorerange = scorerange
        self.uninoise = uninoise
        self.noisefact = noisefact

        assert flat.shape == dark.shape, \
            'Flat and dark are not same shape'
        self.flat = flat
        self.dark = dark

        self.make_image()


    def gaussianpsf(self,flux,xgrid,ygrid,x0,y0,psfhwhm):
        """
        Return a gaussian psf on grid of pixels
        """
        psf = flux * np.exp(-0.5 * ((xgrid-x0) ** 2 + (ygrid-y0) ** 2)
                            / psfhwhm ** 2) / np.sqrt(2. * np.pi * psfhwhm ** 2)
        return psf


    def draw_score(self):
        """
        'score' = (flux)**2/(hwhm)**2/sigma_const**2 
        """
        scoremin = self.scorerange[0]
        scoremax = self.scorerange[1]
        scorerng = scoremax-scoremin
        noscore = True
        while noscore:
            score = np.random.rand() * \
                (scorerng) + scoremin
            cdf   = scoremax * (score-scoremin) / \
                score / (scorerng)
            if 1-cdf > np.random.rand():
                  noscore = False
        return score

    def make_image(self):
        
        self.shape = self.flat.shape
        self.image = np.zeros(self.shape)

        nx, ny = self.shape
        x, y = np.meshgrid(range(nx),range(ny))

        self.score = np.array([])
        for i in range(self.N):
            score = self.draw_score()
            self.score = np.append(self.score,score)
            flux = np.sqrt(score * self.uninoise**2 *
                           self.psfhwhm ** 2)
            x0 = np.random.rand() * (self.shape[0]-1),
            y0 = np.random.rand() * (self.shape[1]-1),
            source = self.gaussianpsf(flux,x,y,x0,y0,self.psfhwhm)
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
    f = FakeField(flat,dark,
                  0.03,
                  0.03,
                  psfhwhm=1.5,
                  Nrange=(15000,30000),
                  scorerange=(15,30000))
    #pl.imshow(f.image,interpolation='nearest')
    #pl.gray()
    #pl.show()

    print f.score
    print f.score.shape

    hdu = pf.PrimaryHDU(f.image)
    hdu.writeto('foo.fits')

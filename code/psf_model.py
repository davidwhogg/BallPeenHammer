import numpy as np
import pyfits as pf

from skimage import transform
from scipy.interpolate import RectBivariateSpline

class PSFModel(object):
    """
    Functions for handing a specified psf model.
    """
    def __init__(self, psf_file, patch_side=5, resolution=4, k=3,
                 scale=None, shear=None):
        assert patch_side % 2 == 1, 'patch shape must be odd and square'
        self.patch_side = patch_side

        # read the model
        self.read_model(psf_file)

        # break the model
        self.break_model(scale, shear)

        # define the model appropriate for the patch
        psf_grid = self.define_model((patch_side, patch_side), resolution)

        # interpolation function
        self.interp = RectBivariateSpline(psf_grid[0], psf_grid[1],
                                          self.psf_model, kx=k, ky=k)

    def read_model(self, psf_file):
        """
        Read the fits file containing the (super-resolution) PSF, assumes
        model is on first HDU.
        """
        f = pf.open(psf_file)
        self.psf_model = f[0].data
        f.close()

    def break_model(self, scale, shear):
        """
        Apply an affine transformation to the model.
        """
        shift_y, shift_x = np.array(self.psf_model.shape[:2]) / 2.
        tf_shift = transform.AffineTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = transform.AffineTransform(translation=[shift_x,
                                                              shift_y])
        tf = transform.AffineTransform(scale=scale, shear=shear)

        self.psf_model = transform.warp(self.psf_model,
                                        (tf_shift + 
                                         (tf + tf_shift_inv)).inverse)

    def define_model(self, patch_shape, resolution):
        """
        Define the grid that the model lives on, truncate if smaller than
        currently defined.
        """
        d = patch_shape[0] * resolution / 2 
        c = (np.array(self.psf_model.shape) - 1) / 2

        self.psf_model = self.psf_model[(c[0] - d):(c[0] + d + 1),
                                        (c[1] - d):(c[1] + d + 1)]

        xg = np.linspace(-0.5 * patch_shape[0], 0.5 * patch_shape[0],
                         patch_shape[0] * resolution + 1)
        yg = np.linspace(-0.5 * patch_shape[1], 0.5 * patch_shape[1],
                         patch_shape[1] * resolution + 1)
        return (xg, yg)

    def render(self, shifts):
        """
        Generate psfs from the model, given a set of shifts
        """
        # handle case where one shift is provided.
        if shifts.shape[1] != 2:
            shifts = np.array([shifts])

        psfs = np.zeros((shifts.shape[0], self.patch_side ** 2.))

        # grid defs for patch
        d = (self.patch_side - 1) / 2.
        g = np.linspace(-d, d, self.patch_side)

        # NOTE - transpose of interp evaluation.
        for i in range(psfs.shape[0]):
            psfs[i] = self.interp(g + shifts[i, 0], g + shifts[i, 1]).T.ravel()

        return psfs / psfs.sum(1)[:, None]

if __name__ == '__main__':
    import matplotlib.pyplot as pl
    model = PSFModel('../psfs/anderson.fits', scale=sc, shear=sh)
    shift = np.array([[0.2, -0.1]])
    bpsfs = model.render(shift)
    pl.gray()
    pl.imshow(psfs.reshape(5,5), interpolation='nearest', origin='lower')
    pl.colorbar()
    pl.show()


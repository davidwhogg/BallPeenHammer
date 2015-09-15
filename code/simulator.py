import numpy as np

from utils import FlatMapper

class FakeProblem(object):
    """
    Generate fake star images, track where they are on the 'detector'.
    """
    def __init__(self, psf_model, flat_rng=0.1, patch_side=5,
                 N_mean=50, flux_range=(50, 5000), bkg_range=(0, 10),
                 detector_side=15, index=2, noise_parms=(0.05, 0.01),
                 bad_frac = 0.05, seed=8675309):
        np.random.seed(seed)

        self.define_layout(N_mean, detector_side, flat_rng)
        self.render_patches(psf_model, patch_side, flux_range, bkg_range,
                            index, noise_parms, bad_frac)

    def define_layout(self, N_mean, detector_side, flat_rng):
        """
        Define the number of stars per pixel and the flat field
`        """
        det_shape = (detector_side, detector_side)

        # stars are poisson distributed across pixels
        self.coverages = np.random.poisson(N_mean, det_shape)
        N = self.coverages.sum()
        self.N = N

        # flat field deviations are uniformly dist, mean one, with specified
        # range
        self.flat_field = np.random.rand(det_shape[0], det_shape[1])
        self.flat_field *= flat_rng
        self.flat_field += 1. - flat_rng / 2

        # record tuple with detector locations, necessary?
        self.pixel_rows = np.zeros(N, np.int)
        self.pixel_cols = np.zeros(N, np.int)
        i, j = 0, 0
        count = 0
        for k in range(N):
            if self.coverages[i, j] == 0:
                count = 0
                i += 1
                if i == detector_side:
                    i = 0
                    j += 1

            self.pixel_rows[k] = i
            self.pixel_cols[k] = j

            count += 1
            if count == self.coverages[i, j]:
                count = 0
                i += 1
                if i == detector_side:
                    i = 0
                    j += 1

    def render_patches(self, psf_model, patch_side, flux_range, bkg_range,
                       index, noise_parms, bad_frac):
        """
        Make fake patches using a psf model, include noise, backgrounds, 
        and a range of flux values.
        """
        # center locations, dont allow less than full patches for now
        buff = (patch_side - 1) / 2
        ind = ((self.pixel_rows >= buff) &
               (self.pixel_rows < self.flat_field.shape[0] - buff) &
               (self.pixel_cols >= buff) &
               (self.pixel_cols < self.flat_field.shape[1] - buff))
        self.row_centers = self.pixel_rows[ind]
        self.col_centers = self.pixel_cols[ind]
        self.N = self.row_centers.size

        # flux and background defs
        self.fluxes = self.draw_fluxes(flux_range, index)
        self.bkgs = np.random.rand(self.N) * (bkg_range[1] - bkg_range[0])
        self.bkgs += bkg_range[0]

        # define subpixel shifts and render psfs
        self.shifts = np.random.rand(self.N, 2) - 0.5
        self.psfs = psf_model.render(self.shifts)

        # flat map
        pls = (self.row_centers, self.col_centers)
        flatmap = FlatMapper(self.flat_field, patch_side, pls)

        # create data
        self.data = self.psfs * self.fluxes[:, None] + self.bkgs[:, None]
        self.true_flats = flatmap.get_1D_flat_patches()
        self.data *= self.true_flats # does this go here??
        self.vars = noise_parms[0] + self.data * noise_parms[1]
        self.data += (np.random.randn(self.N, patch_side ** 2) *
                      np.sqrt(self.vars))

        # make a set of masks to indicate 'bad data'
        size = self.flat_field.size
        bf = np.zeros(size, np.bool)
        Nbad = np.int(np.round(size * bad_frac))
        ind = np.random.permutation(size)[:Nbad]
        bf[ind] = True
        self.bad_field = bf.reshape(self.flat_field.shape)
        badmap = FlatMapper(self.bad_field, patch_side, pls)
        self.masks = badmap.get_1D_flat_patches()

    def draw_fluxes(self, flux_range, index):
        """
        Draw flux values from power law over specified range.
        """
        r = np.random.rand(self.N)
        na = 1. - index
        L = flux_range[0]
        H = flux_range[1]
        return (L ** na + r * (H ** na - L ** na)) ** (1. / na)

if __name__=='__main__':
    from psf_model import PSFModel
    model = PSFModel('../psfs/anderson.fits')
    prob = FakeProblem(model)

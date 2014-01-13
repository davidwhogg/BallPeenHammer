import numpy as np

def make_subsampled(parms, extent, size):
    """
    Make subsampled PSF model using gaussians.

    `parms`: flattened array of amplitudes, x, y, variance_x, variance_y.
             Repeat N times for N gaussians.
    `extent`: extent of model in pixels
    `size`: subsampled size
    """
    xg, yg = np.meshgrid(np.linspace(extent[0], extent[1], size),
                         np.linspace(extent[0], extent[1], size))
    Ngauss = parms.size / 5
    psf = np.zeros((size, size))
    for i in range(Ngauss):
        amp = parms[i * 5]
        x0 = parms[i * 5 + 1]
        y0 = parms[i * 5 + 2]
        vx = parms[i * 5 + 3]
        vy = parms[i * 5 + 4]
        psf += amp * np.exp(-0.5 * ((xg - x0) ** 2 / vx + (yg - y0) ** 2 / vy))
    return psf

def rebin(a, new_shape):
    """
    Return a rebinned array to the new shape.  `new_shape` should be an
    integer factor of the original shape.
    """
    M, N = a.shape
    m, n = new_shape
    new = a.reshape((m, M / m, n, N / n)).sum(3).sum(1)
    return new

if __name__ == '__main__':
    # some reasonable values for WFC3 IR
    wfc_ir_psf = 1.176 / 2. / np.sqrt(2. * np.log(2.)) # in pixels
    extent = np.array([-2.5, 2.5])
    parms = np.array([0.5, 0., 0., wfc_ir_psf ** 2., wfc_ir_psf ** 2.,
                      0.5, -0.5, 0.5, wfc_ir_psf ** 2., wfc_ir_psf ** 2.])

    # test to decide subsample factor
    Nfacs = 100
    tol = 1e-4
    for i in range(Nfacs):
        size = 5 * (Nfacs - i)
        psf = make_subsampled(parms, extent, size=size)
        psf = rebin(psf, (5, 5))
        if i == 0:
            ref = psf
        else:
            err = np.abs(psf - ref)
            if np.any(err > tol):
                print i, size
                break

import numpy as np

def get_grids(data_shape, detector_size, patch_shape, Npsf=(41, 41)):
    """
    Return grid definitions used during fitting
    """
    # random patch centers (actual not needed if not fitting flat)
    # assign positions
    xsize = (data_shape[1] - 1) / 2.
    ysize = (data_shape[2] - 1) / 2.
    xc = np.random.randint(detector_size - 2 * xsize, size=data_shape[0])
    yc = np.random.randint(detector_size - 2 * ysize, size=data_shape[0])
    xc += xsize
    yc += ysize
    patch_centers = (xc, yc)

    # psf_grid defs
    xg = np.linspace(-0.5 * patch_shape[0], 0.5 * patch_shape[0],
                      Npsf[0])
    yg = np.linspace(-0.5 * patch_shape[1], 0.5 * patch_shape[1],
                      Npsf[1])
    psf_grid = (xg, yg)

    # define patch_grid
    yp, xp = np.meshgrid(np.linspace(-ysize, ysize,
                                      data_shape[2]).astype(np.int),
                         np.linspace(-xsize, xsize,
                                      data_shape[1]).astype(np.int))
    patch_grid = (xp, yp)

    return psf_grid, patch_grid, patch_centers

import numpy as np
import pyfits as pf
import matplotlib.pyplot as pl

from matplotlib.colors import LogNorm
from matplotlib.patches import ConnectionPatch
from scipy.interpolate import RectBivariateSpline

def plot(model, center, extent, outname):
    """
    Make a plot demonstrating the generation of a psf from
    the pixel-convolved psf model.
    """
    # define model grid
    xg = np.linspace(-extent, extent, model.shape[0])
    yg = xg.copy()
    interp_func = RectBivariateSpline(xg, yg, model)

    x = np.array([-2, -1, 0, 1, 2]) + center[0]
    y = np.array([-2, -1, 0, 1, 2]) + center[1]
    psf = interp_func(x, y)

    x, y = np.meshgrid(x, y)

    f = pl.figure(figsize=(10, 5))

    pl.gray()
    ax1 = pl.subplot(121)
    ax1.imshow(model, interpolation='nearest', origin='lower',
              extent=(-extent, extent, -extent, extent),
              norm=LogNorm(vmin=model.min(), vmax=model.max()))
    ax1.plot(x, y, 's', mec='r', mfc='none', mew=2)
    pl.axis('off')
    pl.xlim(-2.5, 2.5)
    pl.ylim(-2.5, 2.5)
    ax2 = pl.subplot(122)
    ax2.imshow(psf, interpolation='nearest', origin='lower',
               extent=(-extent, extent, -extent, extent),
               norm=LogNorm(vmin=model.min(), vmax=model.max()))


    coordsA, coordsB = "data", "data"
    pixels = np.array([[0.0, 0.0], [2., 2.], [-1., -1.]])
    locs = np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5]])
    rads = [0.15, 0.25, -0.25]
    for i, p in enumerate(pixels):
        xy1 = p + center
        xy2 = p + locs[i]
        con = ConnectionPatch(xyA=xy2, xyB=xy1, coordsA=coordsA,
                              coordsB=coordsB, axesA=ax2, axesB=ax1,
                              arrowstyle="<-, head_length=1.2, head_width=0.8", 
                              shrinkB=5,
                              connectionstyle='arc3, rad=%s' % rads[i],
                              color='r', lw=2)
        ax2.add_artist(con)
        ax2.plot(p[0], p[1], 's', mfc='none', mec='r', mew=2, ms=50)

    pl.axis('off')
    pl.xlim(-2.5, 2.5)
    pl.ylim(-2.5, 2.5)
    f.savefig(outname)

if __name__ == '__main__':

    f = pf.open('../../psfs/tinytim-pixelconvolved-507-507.fits')
    model = f[0].data
    f.close()

    extent = 2.5
    center = (-0.25, 0.125)
    outname = '../../plots/foo.png'

    plot(model, center, extent, outname)

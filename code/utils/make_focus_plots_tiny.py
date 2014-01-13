import numpy as np
import pyfits as pf
import matplotlib.pyplot as pl

from matplotlib.colors import LogNorm

f = pf.open('../psfs/tinytim/tiny_k4_507_507_-3.fits')
tiny1 = f[0].data
f.close()
f = pf.open('../psfs/tinytim/tiny_k4_507_507_0.fits')
tiny2 = f[0].data
f.close()
f = pf.open('../psfs/tinytim/tiny_k4_507_507_3.fits')
tiny3 = f[0].data
f.close()

a, b = 11, 16
tiny1 = tiny1[a:b, a:b].T
tiny2 = tiny2[a:b, a:b].T
tiny3 = tiny3[a:b, a:b].T


sk = 0.7
f = pl.figure(figsize=(15, 10))
pl.gray()
pl.subplots_adjust(hspace=0.0)

pl.subplot(231)
pl.imshow(tiny1,  interpolation='nearest', origin='lower',
          norm=LogNorm(vmin=tiny1.min(), vmax=tiny1.max()))
pl.title('focus $=\,\,-3.0 \,\mu m$')
mn, mx = tiny1.min(), tiny1.max()
tv = 10. ** np.linspace(np.log10(mn), np.log10(mx), 10)
ts = ['%0.2f' % v for v in tv]
cb = pl.colorbar(shrink=sk, ticks=tv)
cb.ax.set_yticklabels(ts)

pl.subplot(232)
pl.imshow(tiny2,  interpolation='nearest', origin='lower',
          norm=LogNorm(vmin=tiny2.min(), vmax=tiny2.max()))
pl.title('focus $=\,\,0.0 \,\mu m$')
mn, mx = tiny2.min(), tiny2.max()
tv = 10. ** np.linspace(np.log10(mn), np.log10(mx), 10)
ts = ['%0.2f' % v for v in tv]
cb = pl.colorbar(shrink=sk, ticks=tv)
cb.ax.set_yticklabels(ts)
   
pl.subplot(233)
pl.imshow(tiny2,  interpolation='nearest', origin='lower',
           norm=LogNorm(vmin=tiny3.min(), vmax=tiny3.max()))
pl.title('focus $=\,\,3.0 \,\mu m$')
mn, mx = tiny3.min(), tiny3.max()
tv = 10. ** np.linspace(np.log10(mn), np.log10(mx), 10)
ts = ['%0.2f' % v for v in tv]
cb = pl.colorbar(shrink=sk, ticks=tv)
cb.ax.set_yticklabels(ts)

pl.subplot(234)
pl.imshow(tiny1 / tiny2,  interpolation='nearest', origin='lower')
pl.colorbar(shrink=sk)
pl.title('top left divided by top middle')

pl.subplot(236)
pl.imshow(tiny3 / tiny2,  interpolation='nearest', origin='lower')
pl.colorbar(shrink=sk)
pl.title('top right divided by top middle')
f.savefig('../plots/tinytim_focus.png')


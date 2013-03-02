import numpy as np
import matplotlib.pyplot as pl

from scipy.optimize import fmin

np.random.seed(2**4)

def make_center():
    """
    random center, somewhere in or just outside the patch
    """
    x0 = 4*(np.random.rand()-0.5)
    y0 = 4*(np.random.rand()-0.5)
    return x0,y0

def gaussian(f,xg,yg,x0,y0,sig):
    """
    return (double?) gaussian
    """
    frac = 0.0 
    scale = 10
    g1 = f * (1-frac) * np.exp(-0.5*((xg-x0)**2.+(yg-y0)**2.)/sig**2.) / np.sqrt(2.*np.pi*sig**2.)
    g2 = f * (frac) * np.exp(-0.5*((xg-x0)**2.+(yg-y0)**2.)/(sig*10)**2.) / np.sqrt(2.*np.pi*(sig*10)**2.)
    return g1+g2

def make_patch(xg,yg,ff,xc,yc):
    """
    pixelate the gaussian
    """
    os = xc.shape
    xc = xc.ravel() 
    yc = yc.ravel()
    xg = xg.ravel()
    yg = yg.ravel()
    ff = ff.ravel()
    fc = np.zeros(xc.shape[0])
    for i in range(xc.shape[0]):
        ind = (xg>=xc[i]-0.5) & (xg<xc[i]+0.5) & (yg>=yc[i]-0.5) & (yg<yc[i]+0.5) 
        fc[i] = np.mean(ff[ind])
    return fc.reshape(os)

def neglnlike(p,fc,xg,yg,xc,yc,sig,err):
    """
    Negative log likelihood
    """
    mf = gaussian(p[0],xg,yg,p[1],p[2],sig)
    mc = make_patch(xg,yg,mf,xc,yc)
    return np.sum((fc-mc)**2./err**2.)


fwhm = 1.176 # from ISR, at 1.6 micron
sig  = fwhm / (2.*np.sqrt(2.*np.log(2.)))

# gridding
x = np.linspace(-5,5,1000)
y = np.linspace(-5,5,1000)
xg, yg = np.meshgrid(x,y)
x = np.linspace(-2,2,5)
y = np.linspace(-2,2,5)
xc, yc = np.meshgrid(x,y)


# the star
x0, y0 = make_center()
flux = 100.
ff = gaussian(flux,xg,yg,x0,y0,sig)
fc = make_patch(xg,yg,ff,xc,yc)
fc[2,2] *= 0.5
err = fc * 0.05
fc += np.random.randn(5,5) * err

# the model
gss, miter = 0.1, 25
p0 = [flux*(1.+gss*np.random.randn()),
      x0*(1.+gss*np.random.randn()),
      y0*(1.+gss*np.random.randn())]
out = fmin(neglnlike,p0,args=(fc,xg,yg,xc,yc,sig,err),maxiter=miter)
mf = gaussian(out[0],xg,yg,out[1],out[2],sig)
mc = make_patch(xg,yg,mf,xc,yc)

# start fibure
fig = pl.figure()
pl.gray()
ax1 = pl.axes([0.05,0.525,0.45,0.45]) # making by hand since panel
                                      # four sucks
ax1.imshow(fc,interpolation='nearest',vmax=fc.max()*0.2)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax2 = pl.axes([0.5,0.525,0.45,0.45])
ax2.imshow(mc,interpolation='nearest',vmax=fc.max()*0.2)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax3 = pl.axes([0.05,0.025,0.45,0.45])
ratio = fc/mc
ax3.imshow(fc-mc,interpolation='nearest',vmax=fc.max()*0.2)
ax3.set_xticklabels([])
ax3.set_yticklabels([])

# infer for a bunch of patchs
Npatch = 32
flat = np.zeros(5)
exps = np.zeros((Npatch,5))
for i in range(Npatch):
    # making a source
    x0, y0 = make_center()
    flux = 100. * (1+0.1*np.random.randn())
    ff = gaussian(flux,xg,yg,x0,y0,sig)
    fc = make_patch(xg,yg,ff,xc,yc)
    fc[2,2] *= 0.5
    err = fc * 0.05
    fc += np.random.randn(5,5) * err
    # fit it
    p0 = [flux*(1.+gss*np.random.randn()),
          x0*(1.+gss*np.random.randn()),
          y0*(1.+gss*np.random.randn())]
    out = fmin(neglnlike,p0,args=(fc,xg,yg,xc,yc,sig,err),maxiter=miter)
    mf = gaussian(out[0],xg,yg,out[1],out[2],sig)
    mc = make_patch(xg,yg,mf,xc,yc)
    # the flat
    exps[i,:] = fc[2,:] / mc[2,:]
    flat += fc[2,:] / mc[2,:]
flat /= Npatch 

ax4 = pl.axes([0.556,0.025,0.3375,0.45])
for i in range(Npatch):
    pl.plot([-2,-1,0,1,2],exps[i,:],'k',alpha=0.2,drawstyle = 'steps-mid')
ax4.plot([-2,-1,0,1,2],ratio[2,:],'k--',lw=2,drawstyle = 'steps-mid')
ax4.plot([-2,-1,0,1,2],flat,'k',lw=2,drawstyle = 'steps-mid')
ax4.set_xticklabels([])
fig.savefig('toy32.png')












"""
# PSF parms
fwhm = 1.176
sig  = fwhm / (2.*np.sqrt(2.*np.log(2.)))
x0 = (np.random.rand() - 0.5)

# x grids
xc = np.linspace(-5.,5.,11) 
xf = np.linspace(-5,5,1000)

# fine psf
ff = np.exp(-.5*(xf-x0)**2/sig**2.)

# course psf
fc = np.zeros(xc.shape[0])
for i in range(xc.shape[0]):
    ind = (xf>=xc[i]-0.5) & (xf<xc[i]+0.5)
    fc[i] = np.mean(ff[ind])




ax1 = pl.axes([0.1,0.3,0.85,0.6])
ax2 = pl.axes([0.1,0.1,0.85,0.2])

ax1.plot(xf,ff)
ax1.plot(xc,fc,drawstyle='steps-mid')
ax2.plot(xf,ff)
ax2.plot(xc,fc,drawstyle='steps-mid')
pl.xlim(-3,3)
pl.show()

"""

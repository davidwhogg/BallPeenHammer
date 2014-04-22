import numpy as np
import matplotlib.pyplot as pl

from matplotlib.colors import LogNorm, Normalize

def linesearchplot(line_ssqes, line_regs, line_scales, dumpfilebase, iteration):
    """
    Simple evaluation plot of line search at each iteration
    """
    line_regs = np.log(line_regs)
    line_ssqes = np.log(line_ssqes)
    line_costs = line_ssqes + line_regs

    line_scales = np.log(line_scales)
    f = pl.figure()
    pl.plot(line_scales, line_costs, 'k', label='Cost')
    pl.plot(line_scales, line_ssqes, '--k', label='Neg. LogLike')
    pl.plot(line_scales, line_regs, '-.k', label='Regularization')
    pl.xlabel('Ln Step Scale')
    pl.ylabel('Ln Objective')
    pl.legend(loc=2)
    f.savefig(dumpfilebase + '_linesearch_%d.png' % iteration)

def plot_data(i, data, model, bkg, ssqe, old_ssqe, parms, cbscale=0.7):
    """
    Plot data, models, flags, etc.
    """
    # keywords
    kwargs = {'interpolation':'nearest',
              'origin':'lower'}
    clipkwargs = kwargs.copy()
    clipkwargs['vmax'] = parms.clip_parms[1]

    fig= pl.figure(figsize=(12, 16))
    pl.subplots_adjust(wspace=0.05, hspace=0.03)
    pl.figtext(0.075, 0.7, 'Pre-clip', ha='center', va='center',
                rotation='vertical', size=60)
    pl.figtext(0.075, 0.3, 'Post-clip', ha='center', va='center',
                rotation='vertical', size=60)

    d = data.reshape(parms.patch_shape)
    for j in range(2):

        if j == 0:
            ind = parms.flags > 1
            f = parms.flags.copy()
            f[ind] = 0
            s = old_ssqe.reshape(parms.patch_shape)
            b = parms.old_bkg.reshape(parms.patch_shape)
            m = parms.old_model.reshape(parms.patch_shape)
        else:
            b = bkg.reshape(parms.patch_shape)
            s = ssqe.reshape(parms.patch_shape)
            m = model.reshape(parms.patch_shape)
            f = parms.flags.reshape(parms.patch_shape)

        if np.sum(m) == 0.:
            return
            
        logkwargs = kwargs.copy()
        logkwargs['norm'] = LogNorm(vmin=m.min(), vmax=m.max())

        # data, old_model
        pl.subplot(4, 3, 1 + j * 6)
        pl.imshow(np.abs(d), **logkwargs)
        pl.axis('off')
        pl.colorbar(shrink=cbscale)
        pl.title('Log(Data)')
        pl.subplot(4, 3, 2 + j * 6)
        pl.imshow(m, **logkwargs)
        pl.axis('off')
        pl.colorbar(shrink=cbscale)
        pl.title('Log(Model), bkg=%0.3f' % np.mean(bkg))

        # residuals
        ind = f == 0
        r = np.zeros_like(d)
        r[ind] = d[ind] - m[ind]
        pl.subplot(4, 3, 3 + j * 6)
        pl.imshow(r, **kwargs)
        pl.axis('off')
        pl.colorbar(shrink=cbscale)
        pl.title('Data - Model')

        # data flags
        flagkwargs = kwargs.copy()
        flagkwargs['norm'] = Normalize(vmin=0, vmax=3)
        pl.subplot(4, 3, 4 + j * 6)
        pl.imshow(f, **flagkwargs)
        pl.axis('off')
        cb = pl.colorbar(shrink=cbscale, ticks=[0, 1, 2, 3])
        cb.ax.set_yticklabels(['0', '1', '2', '3'])
        pl.title('0=good, 1=mast,\n2=clip, 3=grow')
        
        # clip criterion
        pl.subplot(4, 3, 5 + j * 6)
        var = parms.floor + parms.gain * np.abs(m)
        var += (parms.q * (m - b)) ** 2.
        chi = np.zeros_like(d)
        chi[ind] = np.abs(d[ind] - m[ind]) / np.sqrt(var[ind])
        pl.imshow(chi, **clipkwargs)        
        pl.axis('off')
        pl.colorbar(shrink=cbscale)
        pl.title('$\chi_{clip}\, clip=%0.1f$' % parms.clip_parms[1])

        # Negative Log-Likelihood
        pl.subplot(4, 3, 6 + j * 6)
        pl.imshow(s, **kwargs)
        pl.axis('off')
        pl.colorbar(shrink=cbscale)
        pl.title('Negative Log-likelihood,\ntotal=%0.3f' % s.sum())
        
    fig.savefig(parms.plotfilebase + '_data_%d.png' % parms.data_ids[i])

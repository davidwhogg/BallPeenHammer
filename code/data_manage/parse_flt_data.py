import os
import json
import numpy as np
import pyfits as pf

def json_header(header):
    """
    Make json with header info
    """
    h = {}
    keys = header.keys()
    for i, k in enumerate(keys):
        if k in ['COMMENT', 'PR_INV_L']:
            header[i] = ''.join(header[i].split('\''))
        h[k] = header[i]
    return json.dumps(h)

def load_data(filename):
    """
    Load data, return science data, errors, data quality flags, 
    and a dictionary of json headers.
    """
    f = pf.open(filename)

    sci = f[1].data
    err = f[2].data
    dql = f[3].data

    hds = {}
    keys = ['Primary', 'SCI', 'ERR', 'DQ']
    for i in range(4):
        hds[keys[i]] = json_header(f[i].header)

    f.close()
    return sci, err, dql, hds

def run_source_extractor(sci, err, dql):

    # variances
    shape = err.shape
    err = err.ravel()
    dql = dql.ravel()
    ind = np.in1d(dql, [0, 512]) == False
    err[ind] = np.max(err[ind]) * 1e10
    err = err.reshape(shape)
    dql = dql.reshape(shape)

    # flags
    flags = np.zeros_like(dql)
    ind = dql == 512
    flags[ind] = 512

    # temporary files
    hdu = pf.PrimaryHDU(sci)
    hdu.writeto('data.fits', clobber=True)
    hdu = pf.PrimaryHDU(err ** 2.)
    hdu.writeto('variance.fits', clobber=True)
    hdu = pf.PrimaryHDU(dql * 1000)
    hdu.writeto('flags.fits', clobber=True)

    # run source extractor
    os.system('../../local/sextractor-2.8.6/bin/sex data.fits' +
              ' -c ./data_manage/bph-wfc3-f160w.sex')

def get_patches(data, var, dql, centers, patch_size=5):
    """
    Return patched data, etc, at given centers.
    """

    x, y = np.meshgrid(range(-(patch_size-1)/2, (patch_size-1)/2 + 1),
                       range(-(patch_size-1)/2, (patch_size-1)/2 + 1))

    xcs = (x.ravel()[None, :] + centers[:,0][:, None]).astype(np.int)
    ycs = (y.ravel()[None, :] + centers[:,1][:, None]).astype(np.int)

    test = data[xcs, ycs]
    ind = np.argsort(test, axis=1)
    xshift = x.ravel()[ind[:, -1]]
    yshift = y.ravel()[ind[:, -1]]
    ind = np.where((np.abs(xshift) < 2) & (np.abs(yshift) < 2))[0]
    centers[ind, 0] += xshift[ind]
    centers[ind, 1] += yshift[ind]
    centers = centers[ind]

    xcs = (x[None, :, :] + centers[:,0][:, None, None]).astype(np.int)
    ycs = (y[None, :, :] + centers[:,1][:, None, None]).astype(np.int)

    return data[xcs, ycs], var[xcs, ycs], dql[xcs, ycs], xcs, ycs

def make_ds9_regions(centers):
    """
    DS9 regions for debugging.
    """
    reg = open('foo.reg','w')
    reg.write('global color=green dashlist=8 3 width=1 ' + 
              'font="helvetica 10 normal roman" select=1 ' + 
              'highlite=1 dash=0 fixed=0 edit=1 move=1 ' +
              'delete=1 include=1 source=1\n')
    for i in range(centers.shape[0]):
        reg.write('circle(%0.0f,%0.0f,%d)\n' % (centers[i, 0], 
                                                centers[i, 1], 10))
    reg.close()

def write_fits(patches, filename):
    """
    Write a fits file with patches at different hdus.
    """
    hdus = [pf.PrimaryHDU(np.array([0]))]
    for j in range(patches.shape[0]):
        hdus.append(pf.ImageHDU(patches[j]))
    hdulist = pf.HDUList(hdus)
    hdulist.writeto(filename, clobber=True)


if __name__ == '__main__':

    import warnings
    import matplotlib.pyplot as pl
    warnings.filterwarnings('ignore')

    f = open('list.txt')
    l = f.readlines()
    f.close()

    N = 0
    Ncrowded = 0
    for i, fltfile in enumerate(l):

        sci, err, dql, hds = load_data(fltfile)
        run_source_extractor(sci, err, dql)

        d = np.loadtxt('out.cat')
        ind = (d[:, 1] < 99) & (d[:, -1] > 0.8) & \
            (d[:, 5] > 3) & (d[:, 5] < 1011) & \
            (d[:, 6] > 3) & (d[:, 6] < 1011)
        d = d[ind]

        if (d.shape[0] > 500):
            if Ncrowded == 0:
                pd, pv, pq = get_patches(sci, err**2, dql, d[:, [6, 5]] - 1)
                peaks = pd[:, 2, 2]
                ind = (peaks > 50. * np.median(sci))
                pd, pv, pq = pd[ind], pv[ind], pq[ind]
                write_fits(pd, '../data/crowded_data.fits')
                write_fits(pv, '../data/crowded_var.fits')
                write_fits(pq, '../data/crowded_dq.fits')
                Ncrowded += pq.shape[0]
        else:        
            pd, pv, pq = get_patches(sci, err**2, dql, d[:, [6, 5]] - 1)
            peaks = pd[:, 2, 2]
            ind = (peaks > 50. * np.median(sci))
            pd, pv, pq = pd[ind], pv[ind], pq[ind]
        
            if i == 0:
                data, var, dq = pd, pv, pq
            else:
                data = np.vstack((data, pd))
                var = np.vstack((var, pv))
                dq = np.vstack((dq, pq))
            N += pd.shape[0]
        print i, fltfile[:-1], N, Ncrowded
        if N > 2**8:
            break

    write_fits(data, '../data/test_data.fits')
    write_fits(var, '../data/test_var.fits')
    write_fits(dq, '../data/test_dq.fits')

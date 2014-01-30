import os
import json
import numpy as np

from BallPeenHammer.fitting import fit_single_patch, data_loss
from BallPeenHammer.fitting import update_single_shift
from BallPeenHammer.generation import render_psfs

def load_test_data(metas):
    """
    Load left out (test) data for all trials.
    """
    f = open(metas[0])
    j = json.loads(f.read())
    f.close()

    # The following are the same across the run
    dq = np.loadtxt(j['dqfile'])
    data = np.loadtxt(j['datafile'])
    gain = j['gain']
    floor = j['floor']
    minpixels = j['minpixels']
    background = j['background']

    # get rid of useless patchs
    ind = []
    for i in range(data.shape[0]):
        test = dq[i] == 0
        if np.sum(test) >= minpixels:
            ind.append(i)

    dq = dq[ind]
    data = data[ind]

    test_dq = [None] * len(metas)
    epsilons = np.zeros(len(metas))
    test_data = [None] * len(metas)
    for i in range(len(metas)):
        f = open(metas[i])
        j = json.loads(f.read())
        f.close()
        
        ref = np.arange(data.shape[0]).astype(np.int)
        traininds = np.loadtxt(j['trainindsfile'])
        testinds = np.delete(ref, traininds)
        test_data[i] = data[testinds]

        epsilons[i] = j['eps']

    return test_data, test_dq, epsilons, gain, floor, background

def get_psfs(psffiles, data, dq, detector_size, patch_shape, background, 
             threads, loss_kind, floor, gain, flat=None):
    """
    Build the psfs for the data, given a set of psf and shift files
    """
    if flat is None:
        flat = np.ones((detector_size, detector_size))

    psfs_list = [None] * len(psffiles)
    shifts_list = [None] * len(psffiles)
    for i in range(len(psffiles)):

        # psf model
        f = pf.open(psffiles[i])
        psf_model = f[0].data
        f.close()

        # grid and center defs
        shape = (data.shape[0], patch_shape[0], patch_shape[1])
        psf_grid, patch_grid, patch_centers = get_grid(shape, detector_size,
                                                       patch_shape)

        # compute shifts
        shifts = np.zeros((data.shape[0], 2))
        shifts, ssqe = update_shifts(data, dq, flat, psf_model, psf_grid,
                                     patch_grid, patch_centers, shifts,
                                     background, threads, loss_kind, floor,
                                     gain)

        # render
        psfs[i] = render_psfs(psf_model, shifts, data.shape, psf_grid[0],
                              psf_grid[1])
        
    return psfs, shifts

def cv_score(data, dq, psfs, background, floor, gain, toss_frac=0.005):
    """
    Compute the CV score for the runs.
    """
    flat = np.ones(data[0].size)
    patch_shape = psfs[0][0].shape

    scores = np.zeros(len(psfs))
    for i in range(len(psfs)):
        ssqe = np.zeros(data.shape[0])
        for j in range(len(data.shape[0])):
            ind = dq[j] == 0
            flux, bkg_mod, bkg = fit_single_patch((data[j].ravel(),
                                                   psfs[i][j].ravel(),
                                                   flat, ind.ravel(),
                                                   background))
            model = psfs[i][j] * flux + bkg.reshape(patch_shape)
            ssqe[j] = data_loss(data[j], model, 'nll-model', floor, gain)

        if toss_frac > 0.0:
            Ntoss = np.ciel(data.shape[0] * toss_frac)
            ssqe = np.sort(ssqe)[:-Ntoss]

        scores[i] = np.sum(ssqe)

    return scores

def get_psf_files(metas):
    """
    Get the optimized filenames
    """
    psffiles = []
    for i in range(len(metas)):
        directory = metas[:-10]
        os.system('ls %s/*psf*.fits > foo')
        f = open('foo')
        Niter = len(f.readlines())
        f.close()
        os.system('rm foo')
        psffiles[i] = '%s/%s_psf_%d' % (directory, directory.split('/')[-1],
                                        Niter - 1)
    return psffiles

if __name__ == '__main__':

    run = 1
    threads = 4
    toss_frac = 0.005
    patch_shape = (5, 5)

    os.system('ls ../output/run%d/*json > foo' % run)
    f = open('foo')
    metas = [l[:-1] for l in f.readlines()]
    f.close()
    os.system('rm foo')

    o = load_test_data(metas)
    test_data, test_dq, epsilons, gain, floor, background = *o

    psffiles = get_psf_files(metas)

    psfs_list, shifts_list = get_psfs(psffiles, test_data, test_dq,
                                      detector_size, patch_shape, background,
                                      threads, loss_kind, floor, gain)

    scores = cv_score(test_data, test_dq, psfs, shifts, background,
                      floor, gain, toss_frac=toss_frac)

import numpy as np

from scipy.interpolate import interp1d

def get_hst_focus_models(mjds):
    """
    Return the HST focus model for the given mjds.
    """
    # load the model
    f = open('../data/focus-model-09-13.dat')
    l = f.readlines()
    f.close()
    refmjds = np.zeros(len(l))
    focusmodel = np.zeros(len(l))
    for i, line in enumerate(l):
        line = line.split()
        refmjds[i] = np.float(line[0])
        focusmodel[i] = np.float(line[-1][:-1])

    # interp
    f = interp1d(refmjds, focusmodel)
    ind = (mjds <= refmjds.max())
    models = np.zeros(mjds.size) + np.Inf
    models[ind] = f(mjds[ind])

    return models

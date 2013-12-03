import numpy as np
from scipy.interpolate import RectBivariateSpline

def get_scale(model, data, flux=None):
    """
    Get the appropriate scale for psf model.
    """
    if flux is None:
        d = np.atleast_2d(data.ravel()).T
        m = np.atleast_2d(model.ravel()).T
        rh = np.dot(m.T, d)
        lh = np.dot(m.T, m)
        scale = rh / lh
    else:
        scale = flux

    return (d - m * scale) ** 2., scale

def fit_patch(data, flat, interp_func, steps=(11, 11, 11)):
    """
    Do an adaptive grid search to find best shift and scale
    """
    best_tse = np.inf
    best_shift = [0, 0]
    delta = 0.5
    x = np.array([-2, -1, 0, 1, 2])
    y = np.array([-2, -1, 0, 1, 2])

    for s in steps:

        xs, ys = np.meshgrid(np.linspace(-delta + best_shift[0], 
                                          delta + best_shift[0], s),
                             np.linspace(-delta + best_shift[1], 
                                          delta + best_shift[1], s))
        xs = xs.ravel()
        ys = ys.ravel()
        
        for i in range(xs.size):
            m = interp_func(x + xs[i], y + ys[i]).T
            m *= flat
            
            if s == steps[-1]:
                sqe, flux = get_scale(data, m)
            else:
                flux = np.sum(data)
            
            m *= flux
            tse = np.sum((data - m) ** 2.)
            
            if tse < best_tse:
                best_tse = tse
                best_model = m
                best_shift = (xs[i], ys[i])

        delta = xs[1] - xs[0]

    return best_model, best_shift, best_tse

def fit_known_shift_and_flux(interp_func, shift, flux):
    """
    Return model for known fluxes, shifts
    """
    x = np.array([-2, -1, 0, 1, 2]) + shift[0]
    y = np.array([-2, -1, 0, 1, 2]) + shift[1]
    m = interp_func(x, y).T
    return m * flux

def fit_known_psf_and_shift(data, shifts, ini_flat, xc,
                            xp, yc, yp, tol=0.001, mode='full', 
                            running_flat=None, interp_func=None):
    """
    Fit known psf model and shifts, infer flat
    """
    x = np.array([-2, -1, 0, 1, 2])
    y = np.array([-2, -1, 0, 1, 2])

    tse = np.inf
    flat = ini_flat
    while True:
        norm = np.zeros_like(flat)
        new_tse = 0.0
        new_flat = np.zeros_like(flat)
        for i in range(data.shape[0]):
            psf = interp_func(x + shifts[i][0], y + shifts[i][1]).T
            m = psf * flat[xp + xc[i], yp + yc[i]]
            se, flux = get_scale(m, data[i])
            m *= flux
            norm[xp + xc[i], yp + yc[i]] += 1.
            new_flat[xp + xc[i], yp + yc[i]] += (data[i] / m) * \
                flat[xp + xc[i], yp + yc[i]]
            new_tse += se.sum()
            if np.mod(i, 100) == 0:
                print i

        new_flat /= norm
        new_flat *= new_flat.size / new_flat.sum()
        
        print tse, new_tse
        if (tse - new_tse) / new_tse > tol:
            tse = new_tse
            flat = new_flat
        else:
            if mode == 'sqe':
                running_flat = new_flat
                return new_tse
            else:
                return new_flat, norm, new_tse

def sqe_known_shift(parms, data, shifts, ini_flat, xc,
                    xp, yc, yp, xg, yg, running_flat, eps=0., tol=0.01):
    """
    Return squared error, update running flat
    """
    msize = np.sqrt(parms.size)
    psf_model = parms.reshape(msize, msize)
    interp_func = RectBivariateSpline(xg, yg, psf_model)

    x = np.array([-2, -1, 0, 1, 2])
    y = np.array([-2, -1, 0, 1, 2])

    tse = np.inf
    flat = ini_flat.copy()
    for j in range(2):
        norm = np.zeros_like(flat)
        new_tse = 0.0
        new_flat = np.zeros_like(flat)
        for i in range(data.shape[0]):
            psf = interp_func(x + shifts[i][0], y + shifts[i][1]).T
            m = psf * flat[xp + xc[i], yp + yc[i]] * data[i].sum()
            norm[xp + xc[i], yp + yc[i]] += 1.
            new_flat[xp + xc[i], yp + yc[i]] += (data[i] / m) * \
                flat[xp + xc[i], yp + yc[i]]
            new_tse += np.sum((data[i] - m) ** 2.)

        new_flat /= norm
        new_flat *= new_flat.size / new_flat.sum()
        flat = new_flat

    running_flat = new_flat
    grad = np.gradient(psf_model)
    cost = new_tse + eps * (np.sum(np.abs(grad[0])) + 
                                    np.sum(np.abs(grad[1])))
    print 'Cost =', cost
    return cost


def fit_known_shift(data, shifts, ini_flat, ini_psf, xc,
                    xp, yc, yp, xg, yg, tol=0.001):
    """
    Fit known shifts, infer flat and psf
    """
    running_flat = np.ones_like(ini_flat)
    result = fmin_bfgs(fit_known_psf, )

def fit_known_psf(data, interp_func, ini_flat, xc, xp, yc, yp, tol=0.01):

    tse = np.inf
    flat = ini_flat
    while True:
        norm = np.zeros_like(flat)
        new_tse = 0.0
        new_flat = np.zeros_like(flat)
        for i in range(data.shape[0]):
            m, s, se = fit_patch(data[i], flat[xp + xc[i], yp + yc[i]],
                                 interp_func)
            norm[xp + xc[i], yp + yc[i]] += 1.
            new_flat[xp + xc[i], yp + yc[i]] += (data[i] / m) * \
                flat[xp + xc[i], yp + yc[i]]
            new_tse += se
            if np.mod(i, 100) == 0:
                print i

        new_flat /= norm
        print tse, new_tse
        if (tse - new_tse) / new_tse > tol:
            tse = new_tse
            flat = new_flat
        else:
            return new_flat, norm, new_tse

"""
    Alternate block for sqe_known_shift


    while True:
        norm = np.zeros_like(flat)
        new_tse = 0.0
        new_flat = np.zeros_like(flat)
        for i in range(data.shape[0]):
            psf = interp_func(x + shifts[i][0], y + shifts[i][1]).T
            m = psf * flat[xp + xc[i], yp + yc[i]]
            se, flux = get_scale(m, data[i])
            m *= flux
            norm[xp + xc[i], yp + yc[i]] += 1.
            new_flat[xp + xc[i], yp + yc[i]] += (data[i] / m) * \
                flat[xp + xc[i], yp + yc[i]]
            new_tse += se.sum()

        if (tse - new_tse) / new_tse > tol:
            tse = new_tse
            flat = new_flat
        else:
            running_flat = new_flat
            grad = np.gradient(psf_model)
            cost = new_tse + eps * (np.sum(np.abs(grad[0])) + 
                                    np.sum(np.abs(grad[1])))
            print 'Cost =', cost
            return cost

"""

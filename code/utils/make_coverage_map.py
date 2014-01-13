import psycopg2
import numpy as np
import pyfits as pf

N = 1014
cov = np.zeros((N, N))
yg = np.arange(N)

db = psycopg2.connect('dbname=f160w')
cr = db.cursor()

cmd = 'SELECT x FROM patch_meta ' + \
    'WHERE y >= %d AND y <= %d'
for i in range(N):
    print i
    yn, yx = yg[i] - 2, yg[i] + 2
    cr.execute(cmd % (yn, yx))
    row = np.array(cr. fetchall()).ravel()
    if row.size == 0:
        continue
    for j in range(5):
        bins = np.arange(-2, N, 5) - 0.5 + j
        bins = np.append(bins, bins.max() + 5)
        inds = (bins[:-1] + 2.5).astype(np.int)
        idx  = (inds <= N - 1)
        inds = inds[idx]
        idx  = (bins <= N - 1 + 2.5)
        bins = bins[idx]
        counts, b = np.histogram(row, bins=bins)
        cov[i, inds] = counts

db.close()
hdu = pf.PrimaryHDU(cov.reshape(N, N))
hdu.writeto('../f160w_coverage.fits', clobber=True)

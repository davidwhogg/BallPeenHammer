import warnings
warnings.filterwarnings('ignore')

from parse_flt_data import *
from db_utils import *

f = open('list.txt')
l = f.readlines()
f.close()

os.system('createdb foo')
create_table('foo', 'pixels')
create_table('foo', 'var')
create_table('foo', 'dq')
create_table('foo', 'patch_meta')
create_table('foo', 'image_meta')

keys = ['Primary', 'SCI', 'ERR', 'DQ']

N = 0
Ncrowded = 0
l = l[:100]
for i, fltfile in enumerate(l):

    # load data run sextractor
    sci, err, dql, hds = load_data(fltfile)
    run_source_extractor(sci, err, dql)

    # load and manipulate sextractor catalog
    d = np.loadtxt('out.cat')
    
    # limit data to unblended stars, within bounds
    ind = (d[:, 1] < 99) & (d[:, -1] > 0.8) & \
        (d[:, 5] > 3) & (d[:, 5] < 1011) & \
        (d[:, 6] > 3) & (d[:, 6] < 1011) & \
        (d[:, 9] == 0)
    d = d[ind]

    # patchify
    pd, pv, pq = get_patches(sci, err**2, dql, d[:, [6, 5]] - 1)
    peaks = pd[:, 2, 2]

    # magic number to limit number of faint sources recorded
    ind = (peaks > 25. * np.median(sci))
    peaks = peaks[ind]
    pd, pv, pq = pd[ind], pv[ind], pq[ind]

    print i, fltfile, pd.shape[0]

    # insert data, var, dq, patch_meta
    for jj in range(pd.shape[0]):
        try:
            assert peaks[jj] == np.max(pd[jj])
        except:
            c = d[:, [5, 6]]
            c = c[ind]
            print c[jj]
            print pd[jj]
            print peaks[jj]
            assert 0
        insert_into_table('pixels', pd[jj].ravel().astype(np.str))
        insert_into_table('var', pv[jj].ravel().astype(np.str))
        insert_into_table('dq', pq[jj].ravel().astype(np.str))
        meta = [np.str(np.int(d[jj, 5])),
                np.str(np.int(d[jj, 6])),
                np.str(peaks[jj]),
                '\'' + fltfile + '\'']
        insert_into_table('patch_meta', meta)

    fltfile = fltfile[:-1].split('/')[2]
    # image meta
    meta = ['\'' + fltfile + '\'',
            '\'' + hds[keys[0]] + '\'',
            '\'' + hds[keys[1]] + '\'',
            '\'' + hds[keys[2]] + '\'',
            '\'' + hds[keys[3]] + '\'',
            np.str(pd.shape[0])]
    insert_into_table('image_meta', meta)

import warnings
warnings.filterwarnings('ignore')

from parse_flt_data import *
from db_utils import *

dbname = 'f160w'
keys = ['Primary', 'SCI', 'ERR', 'DQ']

# create tables if necessary
create_table(dbname, 'pixels')
create_table(dbname, 'var')
create_table(dbname, 'dq')
create_table(dbname, 'persist')
create_table(dbname, 'patch_meta')
create_table(dbname, 'image_meta')

# loop over chunks and stuff it.
N = 44
for i in range(N):
    
    os.system('ls ../data/flt/chunk%d/*flt.fits > foo.txt' % i)
    f = open('foo.txt')
    l = f.readlines()
    f.close
    
    for fltfile in l:
        

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
        pd, pv, pq, xs, ys = get_patches(sci, err**2, dql, d[:, [6, 5]] - 1)
        peaks = pd[:, 2, 2]

        # magic number to limit number of faint sources recorded
        ind = (peaks > 25. * np.median(sci))
        peaks = peaks[ind]
        pd, pv, pq = pd[ind], pv[ind], pq[ind]
        xs, ys, d = xs[ind], ys[ind], d[ind]

        # get persistence
        f = pf.open(fltfile)
        propid = f[0].header['PROPOSID']
        f.close()

        os.system('ls ../data/persist/%d/*persist.fits > foo.txt' % propid)
        f = open('foo.txt')
        persists = f.read().split('_persist.fits\n')
        persists = ''.join(persists).split('../data/persist/%d/' % propid)
        f.close

        if fltfile.split('/')[-1].split('_')[0] in persists:
            f = pf.open('../data/persist/%d/' % propid + 
                        fltfile.split('/')[-1][:-10] +
                        '_persist.fits')
            persist = f[1].data
            f.close()
        else:
            persist = np.zeros_like(sci)

        pp = persist[xs, ys]

        # insert data, var, dq, patch_meta
        pd = pd.reshape(pd.shape[0], pd.shape[1] ** 2.)
        pv = pv.reshape(pv.shape[0], pv.shape[1] ** 2.)
        pq = pq.reshape(pq.shape[0], pq.shape[1] ** 2.)
        pp = pp.reshape(pp.shape[0], pp.shape[1] ** 2.)
        if pd.shape[0] > 0:
            insert_into_table('pixels', pd.astype(np.str), dbname)
            insert_into_table('var', pv.astype(np.str), dbname)
            insert_into_table('dq', pq.astype(np.str), dbname)
            insert_into_table('persist', pp.astype(np.str), dbname)
            narr = ['\'' + fltfile + '\'' for j in range(pd.shape[0])]
            meta = np.array([d[:, 5].astype(np.int).astype(np.str),
                             d[:, 6].astype(np.int).astype(np.str),
                             peaks.astype(np.str),
                             narr]).T
            insert_into_table('patch_meta', meta, dbname)
            meta = np.array([['\'' + fltfile + '\'', 
                              propid, np.str(pd.shape[0])]])
            insert_into_table('image_meta', meta, dbname)

        print i, fltfile, pd.shape[0]

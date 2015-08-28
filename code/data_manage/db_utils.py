import sys
import string
import psycopg2
import numpy as np
import pyfits as pf

sys.path.append('../../local/')
import pyflann

def create_table(dbname, table, Npix=25):
    """
    Create tables in postgres for wfc3ir patches
    """
    assert table in ['pixels', 'var', 'dq', 'persist',
                     'patch_meta', 'image_meta']
    command = 'CREATE TABLE IF NOT EXISTS '

    colnames = get_table_colnames(table, Npix)

    if table in ['pixels', 'var', 'dq', 'persist']:
        if table == 'dq':
            fmt = ' INT, '
        else:
            fmt = ' DOUBLE PRECISION, '
        names = '(id SERIAL, '
        for i in range(Npix):
            names += colnames[i] + fmt
            create = command + table + names[:-2] +')'

    if table == 'patch_meta':
        create = command + 'patch_meta (id SERIAL, x INT, y INT, ' + \
            'peak DOUBLE PRECISION, ra DOUBLE PRECISION, ' + \
            'dec DOUBLE PRECISION, mast_file TEXT)'

    if table == 'image_meta':
        create = command + 'image_meta (mast_file TEXT, ' + \
            'prop_id INT, ' + \
            'Nsrcs SMALLINT)'

    db = psycopg2.connect('dbname=' + dbname)
    cursor = db.cursor()
    cursor.execute(create)
    db.commit()
    db.close()

def get_table_colnames(table, Npix):
    """
    Return list of column names for table
    """
    if table == 'pixels':
        colnames = ['pix%d' % i for i in range(Npix)]
    if table == 'var':
        colnames = ['var%d' % i for i in range(Npix)]
    if table == 'dq':
        colnames = ['dq%d' % i for i in range(Npix)]
    if table == 'persist':
        colnames = ['per%d' % i for i in range(Npix)]
    if table == 'patch_meta':
        colnames = ['x', 'y', 'peak', 'ra', 'dec', 'mast_file']
    if table == 'image_meta':
        colnames = ['mast_file', 'prop_id', 'Nsrcs']
    return colnames

def insert_into_table(table, data, dbname, Npix):
    """
    Insert data into table, data is numpy array.
    """
    s = '('
    for i in range(data.shape[1]):
        s += '%s,'
    s = s[:-1] + ')'

    db = psycopg2.connect('dbname=' + dbname)
    cursor = db.cursor()

    arg = ','.join(cursor.mogrify(s, x) for x in data)

    colnames = get_table_colnames(table, Npix)
    insert_head = 'INSERT INTO ' + table + ' (' + \
        string.join(colnames, ',') + ') VALUES '
    insert = insert_head + arg

    cursor.execute(insert)
    db.commit()
    db.close()

def get_data(xmin, xmax, ymin, ymax, dbname):

    # data to grab, other than patch_meta
    keys = ['pixels', 'dq', 'var', 'persist']

    # connect
    db = psycopg2.connect('dbname=%s' % dbname)
    cr = db.cursor()
    data = {}

    # need patch_meta to get positions'
    cmd = 'SELECT * FROM patch_meta ' + \
        'WHERE patch_meta.y >= %d AND patch_meta.y <= %d AND ' + \
        'patch_meta.x >= %d AND patch_meta.x <= %d'
    cr.execute(cmd % (ymin, ymax, xmin, xmax))

    data['patch_meta'] = np.array(cr.fetchall())

    for i in range(len(keys)):
        # get data via id
        cmd = 'SELECT * FROM ' + keys[i] + \
            ' WHERE id = any(array['
        cmd += ','.join(data['patch_meta'][:, 0]) + '])'
        cr.execute(cmd) 
        
        # sort along ids, eliminate id column
        tmp = np.array(cr.fetchall()).astype(np.float)
        ind = np.argsort(tmp[:, 0])
        data[keys[i]] = tmp[ind, 1:]

    db.close()

    return data

def get_data_with_source_ids(xmin, xmax, ymin, ymax, dbname, pathbase, tol=0.5):
    """
    Find all patches within a given region, identifying repeat observations.
    """
    # arcsec to degrees.
    tol /= 3600.

    # connect
    db = psycopg2.connect('dbname=%s' % dbname)
    cr = db.cursor()
    data = {}

    # first get image and visit names.
    cmd = 'SELECT mast_file FROM patch_meta ' + \
        'WHERE patch_meta.y >= %d AND patch_meta.y <= %d AND ' + \
        'patch_meta.x >= %d AND patch_meta.x <= %d'
    cr.execute(cmd % (ymin, ymax, xmin, xmax))
    l = cr.fetchall()
    imglist = []
    visitlist = []
    for i, img in enumerate(l):
        if len(img[0]) == 40:
            imglist.append(img[0][20:-2])
            visitlist.append(img[0][20:26])
        else:
            imglist.append(img[0][21:-2])
            visitlist.append(img[0][21:27])
    imglist = np.unique(np.array(imglist))
    visitlist = np.unique(np.array(visitlist))

    # for each visit, get all patches and build array with x, y, ra, dec,
    # image id.  Source match, assign ids
    Nsrcs = 0
    count = 0
    for i in range(visitlist.size):
        # two queries due to length of filenames
        cmd = 'SELECT * FROM %s ' + \
            'WHERE substring(mast_file from %d for 6) = \'%s\' AND y >= %d ' + \
            'AND y <= %d AND x >= %d AND x <= %d'
        data = []
        starts = [21, 22]
        for start in starts:
            cr.execute(cmd % ('patch_meta', start, visitlist[i], ymin, ymax,
                              xmin, xmax))
            data.extend(cr.fetchall())

        # construct numpy array, find unique peak fluxes
        visit_meta = []
        for j in range(len(data)):
            visit_meta.append([data[j][1], data[j][2], data[j][3], data[j][4],
                               data[j][5], data[j][6][-20:], data[j][0]])
        visit_meta = np.array(visit_meta)
        foo, ind = np.unique(visit_meta[:, 2], return_index=True)
        visit_meta = visit_meta[ind]
        if i == 0:
            meta_data = visit_meta.copy()
        else:
            meta_data = np.vstack((meta_data, visit_meta))

        # assign source ids
        ras = visit_meta[:, 3].astype(np.float)
        decs = visit_meta[:, 4].astype(np.float)
        names = visit_meta[:, -2]
        fluxes = visit_meta[:, 2].astype(np.float)
        sids = np.zeros((visit_meta.shape[0], 1), dtype=np.int) - 99
        Nobs = np.unique(names).size
        Nsrcs += visit_meta.shape[0]
        for j in range(visit_meta.shape[0]):
            if sids[j] == -99:
                sids[j] = count
                dist = np.sqrt((ras[j] - ras) ** 2. + (decs[j] - decs) ** 2.)
                ind = np.where((dist < tol) & (dist > 0.))[0]
                if ind.size > Nobs:
                    sids[j] = -99
                else:
                    sids[ind] = count
                    count = 1 + count
        if i == 0:
            source_ids = sids
        else:
            source_ids = np.vstack((source_ids, np.atleast_2d(sids)))

    # sort database ids
    ind = np.argsort(meta_data[:, -1].astype(np.int))
    meta_data = meta_data[ind]
    source_ids = source_ids[ind]

    # get pixels
    cmd = 'SELECT * FROM pixels WHERE id = any(array['
    cmd += ','.join(meta_data[:, -1]) + '])'
    cr.execute(cmd) 
    tmp = np.array(cr.fetchall()).astype(np.float)
    pixels = tmp[:, 1:]

    # get persistence and subtract
    cmd = 'SELECT * FROM persist WHERE id = any(array['
    cmd += ','.join(meta_data[:, -1]) + '])'
    cr.execute(cmd) 
    tmp = np.array(cr.fetchall()).astype(np.float)
    pixels -= tmp[:, 1:]

    # get dq flags
    cmd = 'SELECT * FROM dq WHERE id = any(array['
    cmd += ','.join(meta_data[:, -1]) + '])'
    cr.execute(cmd) 
    tmp = np.array(cr.fetchall()).astype(np.float)
    dq = tmp[:, 1:]

    db.close()

    # write the query
    h = pf.PrimaryHDU(pixels)
    h.writeto(pathbase + '_pixels.fits', clobber=True)
    h = pf.PrimaryHDU(dq)
    h.writeto(pathbase + '_dq.fits', clobber=True)
    cols = pf.ColDefs([pf.Column(name='x', format='J',
                                 array=meta_data[:, 0].astype(np.int)),
                       pf.Column(name='y', format='J',
                                 array=meta_data[:, 1].astype(np.int)),
                       pf.Column(name='peak', format='D',
                                 array=meta_data[:, 2].astype(np.float)),
                       pf.Column(name='ra', format='D',
                                 array=meta_data[:, 3].astype(np.float)),
                       pf.Column(name='dec', format='D',
                                 array=meta_data[:, 4].astype(np.float)),
                       pf.Column(name='file', format='50A',
                                 array=meta_data[:, 5]),
                       pf.Column(name='db_id', format='K',
                                 array=meta_data[:, 6].astype(np.int)),
                       pf.Column(name='source_id', format='J',
                                 array=source_ids.astype(np.int))])
    t = pf.new_table(cols)
    t.writeto(pathbase + '_meta.fits', clobber=True)

def get_proposal_obs(dbname, propid, pathbase, patch_size=81, tol=0.5):
    """
    Get all the patches associated with a proposal id.
    """
    # connect
    db = psycopg2.connect('dbname=%s' % dbname)
    cr = db.cursor()
    data = {}

    # first get image names for the propid.
    cmd = 'SELECT mast_file FROM image_meta ' + 'WHERE image_meta.prop_id = %s'
    cr.execute(cmd % propid)
    l = cr.fetchall()

    # get source meta data that match images.
    cmd = 'SELECT * FROM patch_meta ' + 'WHERE patch_meta.mast_file = %s'

    # get the patch meta data
    meta_data = []
    for img in l:
        if len(img[0]) == 40:
            start = 20
        else:
            start = 21
        visit = img[0][start:start + 9]
            
        cmd = 'SELECT * FROM patch_meta ' + \
              'WHERE substring(mast_file from %d for 9) = \'%s\' '

        cr.execute(cmd % (start + 1, visit))
        meta_data.extend(cr.fetchall())

    # unpack
    N = len(meta_data)
    data['db_ids'] = np.zeros(N, np.int)
    data['ra-decs'] = np.zeros((N, 2))
    data['x-ys'] = np.zeros((N, 2))
    for i in range(len(meta_data)):
        d = meta_data[i]
        data['db_ids'][i] = d[0]
        data['x-ys'][i] = d[1:3]
        data['ra-decs'][i] = d[4:6]

    # nearest neighbors
    flann = pyflann.FLANN()
    parms = flann.build_index(data['ra-decs'], target_precision=1.0,
                              log_level='info')
    inds, dists = flann.nn_index(data['ra-decs'], len(l),
                                 check=parms['checks'])

    # get data and assign source labels
    sid = 0
    data['source_ids'] = np.zeros(N, np.int) - 99
    data['pixels'] = np.zeros((N, patch_size))
    data['dq'] = np.zeros((N, patch_size), np.int)
    for i in range(N):
        if i % 500 == 0: print i, sid
        if data['source_ids'][i] == -99:
            idx = np.where(dists[i] < (tol / 3600.) ** 2.)[0]
            if idx.size > 1:
                tmp = ['%d' % v for v in data['db_ids'][inds[i, idx]]]
                cmd = 'SELECT * FROM %s WHERE id = any(array['
                cmd += ','.join(tmp) + '])'
                cr.execute(cmd % 'pixels')
                pixels = np.array(cr.fetchall()).astype(np.float)[:, 1:]
                tmp = np.sum(pixels, 1)
                cr.execute(cmd % 'persist')
                pixels -= np.array(cr.fetchall()).astype(np.float)[:, 1:]
                assert (np.std(tmp) / np.mean(tmp) < 0.5)
                cr.execute(cmd % 'dq')
                dq = np.array(cr.fetchall()).astype(np.float)[:, 1:]
                data['source_ids'][inds[i, idx]] = sid
                data['pixels'][inds[i, idx]] = pixels
                data['dq'][inds[i, idx]] = dq
                sid += 1

    # write the query
    h = pf.PrimaryHDU(data['pixels'])
    h.writeto(pathbase + '_pixels.fits', clobber=True)
    h = pf.PrimaryHDU(data['dq'])
    h.writeto(pathbase + '_dq.fits', clobber=True)
    cols = pf.ColDefs([pf.Column(name='x', format='J',
                                 array=data['x-ys'][:, 0].astype(np.int)),
                       pf.Column(name='y', format='J',
                                 array=data['x-ys'][:, 0].astype(np.int)),
                       pf.Column(name='ra', format='D',
                                 array=data['ra-decs'][:, 0].astype(np.float)),
                       pf.Column(name='dec', format='D',
                                 array=data['ra-decs'][:, 0].astype(np.float)),
                       pf.Column(name='db_id', format='K',
                                 array=data['db_ids']),
                       pf.Column(name='source_ids', format='J',
                                 array=data['source_ids'])])
    t = pf.new_table(cols)
    t.writeto(pathbase + '_meta.fits', clobber=True)

if __name__ == '__main__':

    mn, mx = 457, 557
    dbase = 'f160w_9'

    get_proposal_obs(dbase, '12696', 'prop12696_matched')
    


    """
    filebase = '../data/region/f160w_25_%d_%d_%d_%d' % (mn, mx, mn, mx)

    get_data_with_source_ids(mn, mx, mn, mx, dbase, filebase)
    """

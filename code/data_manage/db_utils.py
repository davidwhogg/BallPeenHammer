import string
import psycopg2
import numpy as np

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

def get_matching_data(propids, dbname, tol=1.):
    """
    Find all repeat observations of stars using a restricted
    list of proposal IDs.
    """
    # arcsec to degrees.
    tol /= 3600.

    # connect
    db = psycopg2.connect('dbname=%s' % dbname)
    cr = db.cursor()
    data = {}

    # first get image names.
    imglist = []
    for p in propids:
        cmd = 'SELECT mast_file FROM image_meta ' + \
            'WHERE prop_id = %s' % p
        cr.execute(cmd)
        imglist.extend(cr.fetchall())

    print imglist
    # run through list, get ra, dec
    ras = []
    decs = []
    for img in imglist:
        cmd = 'SELECT ra FROM patch_meta ' + \
            'WHERE mast_file = %s' % img
        cr.execute(cmd)
        ras.extend(cr.fetchall())
        print img
        print cmd
        print ras
        assert 0
    
    print ras
    print len(ras)

    db.close()

if __name__ == '__main__':

    pid = ['11099']
    dbase = 'f160w_25'

    get_matching_data(pid, dbase)

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

    colnames = get_table_colnames(table)

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
            'peak DOUBLE PRECISION, mast_file TEXT)'

    if table == 'image_meta':
        create = command + 'image_meta (mast_file TEXT, ' + \
            'prop_id INT, ' + \
            'Nsrcs SMALLINT)'

    db = psycopg2.connect('dbname=' + dbname)
    cursor = db.cursor()
    cursor.execute(create)
    db.commit()
    db.close()

def get_table_colnames(table):
    """
    Return list of column names for table
    """
    if table == 'pixels':
        colnames = ['pix%d' % i for i in range(25)]
    if table == 'var':
        colnames = ['var%d' % i for i in range(25)]
    if table == 'dq':
        colnames = ['dq%d' % i for i in range(25)]
    if table == 'persist':
        colnames = ['per%d' % i for i in range(25)]
    if table == 'patch_meta':
        colnames = ['x', 'y', 'peak', 'mast_file']
    if table == 'image_meta':
        colnames = ['mast_file', 'prop_id', 
                    'Nsrcs']
    return colnames

def insert_into_table(table, data, dbname):
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

    colnames = get_table_colnames(table)
    insert_head = 'INSERT INTO ' + table + ' (' + \
        string.join(colnames, ',') + ') VALUES '
    insert = insert_head + arg

    cursor.execute(insert)
    db.commit()
    db.close()

def get_data(xmin, xmax, ymin, ymax):

    # data to grab, other than patch_meta
    keys = ['pixels', 'dq', 'var', 'persist']

    # connect
    db = psycopg2.connect('dbname=f160w')
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

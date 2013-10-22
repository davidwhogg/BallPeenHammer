import string
import psycopg2

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

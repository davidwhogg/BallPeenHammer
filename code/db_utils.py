import string
import psycopg2

def create_table(dbname, table, Npix=25):
    """
    Create tables in postgres for wfc3ir patches
    """
    assert table in ['pixels', 'var', 'dq', 'patch_meta', 
                     'image_meta']
    command = 'CREATE TABLE IF NOT EXISTS '

    colnames = get_table_colnames(table)

    if table in ['pixels', 'var', 'dq']:
        names = '(id SERIAL, '
        for i in range(Npix):
            names += colnames[i] + ' REAL, '
            create = command + table + names[:-2] +')'

    if table == 'patch_meta':
        create = command + 'patch_meta (id SERIAL, x INT, y INT, ' + \
            'mast_name TEXT)'

    if table == 'image_meta':
        create = command + 'image_meta (mast_name TEXT, header TEXT, ' + \
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
    if table == 'patch_meta':
        colnames = ['x', 'y', 'mast_name']
    if table == 'image_meta':
        colnames = ['mast_name', 'header', 'Nsrcs']
    return colnames

def insert_into_table(table, data):
    """
    Insert data into table.
    """
    colnames = get_table_colnames(table)

    insert_head = 'INSERT INTO ' + table + ' (' + \
        string.join(colnames, ',') + ') VALUES ('
    insert = insert_head + string.join(data, ',') + ')'
    db = psycopg2.connect('dbname=foo')
    cursor = db.cursor()
    cursor.execute(insert)
    db.commit()
    db.close()

if __name__ == '__main__':

    import os
    import numpy as np

    # example creation and insertion
    os.system('createdb foo')
    create_table('foo', 'pixels')
    create_table('foo', 'var')
    create_table('foo', 'dq')
    create_table('foo', 'patch_meta')
    create_table('foo', 'image_meta')

    db = psycopg2.connect('dbname=foo')
    cursor = db.cursor()
    cursor.execute("""SELECT table_name FROM information_schema.tables 
       WHERE table_schema = 'public'""")

    rows = cursor.fetchall()
    print '\nTables:\n', rows

    pix = np.random.randn(25).astype(np.str)
    insert_into_table('pixels', pix)

    db = psycopg2.connect('dbname=foo')
    cursor = db.cursor()
    cursor.execute("""SELECT * from pixels""")

    rows = cursor.fetchall()
    print '\nTable - pixels:\n', rows

    var = (np.random.randn(25) ** 2).astype(np.str)
    insert_into_table('var', var)

    db = psycopg2.connect('dbname=foo')
    cursor = db.cursor()
    cursor.execute("""SELECT * from var""")

    rows = cursor.fetchall()
    print '\nTable - var:\n', rows

    dq = np.arange(25, dtype=np.int).astype(np.str)
    insert_into_table('dq', dq)

    db = psycopg2.connect('dbname=foo')
    cursor = db.cursor()
    cursor.execute("""SELECT * from dq""")

    rows = cursor.fetchall()
    print '\nTable - dq:\n', rows

    meta = ['235', '675', '\'llsbous39\'']
    insert_into_table('patch_meta', meta)

    db = psycopg2.connect('dbname=foo')
    cursor = db.cursor()
    cursor.execute("""SELECT * from patch_meta""")

    rows = cursor.fetchall()
    print '\nTable - patch_meta:\n', rows

    meta = ['\'llsbous39\'', '\'this will be the header info in json\'', '204']
    insert_into_table('image_meta', meta)

    db = psycopg2.connect('dbname=foo')
    cursor = db.cursor()
    cursor.execute("""SELECT * from image_meta""")

    rows = cursor.fetchall()
    print '\nTable - image_meta:\n', rows

    db.close()
    os.system('dropdb foo')

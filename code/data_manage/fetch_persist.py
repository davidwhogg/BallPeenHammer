import os
import json
import time
import string
import subprocess
import numpy as np
import pyfits as pf

from get_data_from_mast import *
from parse_flt_data import *
from urllib2 import urlopen
#from db_utils import *
from time import time, sleep

def get_persistence_files(prop_id, visit):
    persist_url = 'http://archive.stsci.edu/pub/wfc3_persist/'
    
    filename = prop_id + '.Visit' + visit + '.tar.gz'
    tail = [prop_id, 'Visit' + visit, filename]
    dest = persist_url + '/'.join(tail)
    subprocess.check_output(['wget', dest])

def move_and_manage_persist(prop_id):

    od = os.getcwd()
    os.system('mv *%s*gz ../data/persist/%s' % (prop_id, prop_id))
    os.chdir('../data/persist/' + prop_id)

    os.system('ls > foo.txt')
    f = open('foo.txt')
    l = f.readlines()
    f.close()

    for i in range(len(l)):
        if l[i][-3:] == 'gz\n':
            os.system('tar -xvf ' + l[i])
            os.system('mv %s/*fits .' % l[i][:-8])
            os.system('rm -rf %s' % l[i][:-8])
    os.system('rm *gz')

    os.system('ls > foo.txt')
    f = open('foo.txt')
    l = f.readlines()
    f.close()

    for i in range(len(l)):
        if l[i][-5:] == 'fits\n':
            f = pf.open(l[i])
            if f[0].header['FILTER'].rstrip() != 'F160W':
                os.system('rm ' + l[i])
            f.close()
    os.system('rm *txt')
    os.chdir(od)

f = open('../data/f160w_persist.json')
d = json.load(f)
f.close()

Nvisits = len(d)
visits = np.array([d[i]['Visit'].lower() for i in range(Nvisits)])
prop_ids = np.array([d[i]['Proposal ID'] for i in range(Nvisits)])
filters = np.array([d[i]['Filter'] for i in range(Nvisits)])
tmp = np.array([d[i]['Visit'] + d[i]['Proposal ID'] for i in range(Nvisits)])
aper = np.array([d[i]['Aper'] for i in range(Nvisits)])
digits = np.array([np.int(d[i]['Proposal ID']) for i in range(Nvisits)])

ind = (aper == 'IR') | (aper == 'IR-FIX') | (aper == 'IR-UVIS-FIX')
visits = visits[ind]
prop_ids = prop_ids[ind]
filters = filters[ind]
tmp = tmp[ind]
digits = digits[ind]
Nvisits = visits.size
print Nvisits, tmp.size, digits.size, filters.size, visits.size, prop_ids.size

ind = (filters == 'F160W')
visits = visits[ind]
prop_ids = prop_ids[ind]
filters = filters[ind]
tmp = tmp[ind]
digits = digits[ind]
Nvisits = visits.size
print Nvisits, tmp.size, digits.size, filters.size, visits.size, prop_ids.size

u, ind = np.unique(tmp, return_index=True)
visits = visits[ind]
prop_ids = prop_ids[ind]
filters = filters[ind]
tmp = tmp[ind]
digits = digits[ind]
Nvisits = visits.size
print Nvisits, tmp.size, digits.size, filters.size, visits.size, prop_ids.size


ind = np.argsort(digits)
visits = visits[ind]
prop_ids = prop_ids[ind]
filters = filters[ind]
tmp = tmp[ind]
digits = digits[ind]
Nvisits = visits.size
print Nvisits, tmp.size, digits.size, filters.size, visits.size, prop_ids.size

for i in range(Nvisits):

    os.system('mkdir ../data/persist/' + prop_ids[i])

    # get persistence data from mast
    try:
        get_persistence_files(prop_ids[i], visits[i].upper())
        move_and_manage_persist(prop_ids[i])
    except:
        pass


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
from time import time, sleep


f = open('../data/f160w_list.json')
d = json.load(f)
f.close()

Ndatasets = len(d)
datasets = np.array([d[i]['Dataset'].lower() for i in range(Ndatasets)])

mastuser = sys.argv[1]
mastpass = sys.argv[2]
hostname = sys.argv[3]

setup_cookies()

od = os.getcwd()

Nstep = 100
for i in range(40, Ndatasets):

    dest_dir = '../data/flt/chunk%d' % i
    os.system('mkdir ' + dest_dir)

    dsets = datasets[Nstep * i: Nstep * (i + 1)]
    if len(dsets) == 0:
        break

    # get flt data from mast
    get_datasets(dsets, mastuser, mastpass)

    # was submission successful?
    f = open('res2.html')
    o = f.readlines()
    f.close()
    good = False
    for j in range(len(o)):
        l = '<'.join(o[j].split('>')).split('<')
        if l[0] == 'The request number is ':
            sub_id = l[2]
        if 'Your request was successfully submitted.\n' in l:
            good = True
    assert good, \
        'Failed submission at %d, prop_id %d' % (i, prop_ids[i])
    print dsets
    print i, 'Submitted', sub_id

    # wait until mast posts the data
    url = 'http://archive.stsci.edu/cgi-bin/reqstat?reqnum=='
    waited = 0
    while True:
        print 'Waited:', waited
        t = time()
        sleep(120)
        f = urlopen(url + sub_id)
        o = '<'.join(f.read().split('>')).split('<')
        f.close()
        if 'KILLED' in o:
            assert False, \
                'Killed sftp at %d, prop_id %d' % (i, prop_ids[i])
        if 'complete-succeeded' in o:
            break
        else:
            waited += time() - t
            t = time()
            if waited >= 10 * 3600:
                assert False, \
                    'Failed sftp at %d, prop_id %d' % (i, prop_ids[i])

    sleep(30)
    data_dir = '/stage/' + mastuser + '/' + sub_id
    os.system('ncftpget -R -v -u %s -p %s archive.stsci.edu . ' % \
                  (mastuser, mastpass) + data_dir)
    os.system('mv ' + sub_id + '/*fits ' + dest_dir)
    os.system('rm -rf ' + sub_id)

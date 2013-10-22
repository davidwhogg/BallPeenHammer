#! /usr/bin/env python
#
# Borrowed in part from D. Lang:
# http://astrometry.net/svn/trunk/projects/phat/poll.py
import os
import sys
import urllib2
from urllib import urlencode
from urllib2 import urlopen

def setup_cookies():
    cookie_handler = urllib2.HTTPCookieProcessor()
    opener = urllib2.build_opener(cookie_handler)
    # ...and install it globally so it can be used with urlopen.
    urllib2.install_opener(opener)

def get_datasets(datasets, mastuser, mastpass, datatypes=['FLT']):

    url = 'http://archive.stsci.edu/hst/search.php?action=Search&sci_data_set_name=' + ''.join(datasets)

    f = urlopen(url)
    f.read()

    # Submit the resulting "HST search results" form...
    url = 'http://archive.stsci.edu/cgi-bin/stdads_retrieval_options'

    D = dict(RESULTS_TYPE='science',
             mission='hst')
    D['.submit'] = 'Submit marked data for retrieval from STDADS'
    data = urlencode(D)
    marks = '&'.join([''] + ['stdads_mark=%s' % d.upper() for d in datasets])
    data += marks
    f = urlopen(url, data)
    txt = f.read()
    open('res.html','wb').write(txt)
	
    # Submit the "Retrieval Options" form.

    url = 'https://archive.stsci.edu/cgi-bin/stdads_submit'

    D = dict(PAGE='options',
             archive_username=mastuser,
             archive_password=mastpass,
             media='host',
             destination_hostname='',
             destination_directory='',
             destination_username='',
             destination_password='',
             data='calibrated',
             action='Send retrieval request to ST-DADS',
             specific_extension='')
    data = urlencode(D)
    data += ''.join(['&file_extension=%s' % ext for ext in datatypes])
    data += marks

    f = urlopen(url, data)
    txt = f.read()
    open('res2.html','wb').write(txt)


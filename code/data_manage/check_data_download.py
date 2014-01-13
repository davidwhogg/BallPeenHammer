import os
import json
import pyfits as pf

f = open('../data/f160w_list.json')
fd = json.load(f)
f.close()

print fd[:2]

assert 0
f = open('../data/f160w_persist.json')
pd = json.load(f)
f.close()

N = 44
for i in range(N):
    os.system('ls ../data/flt/chunk%d/*flt.fits > foo.txt' % i)
    f = open('foo.txt')
    l = f.readlines()
    f.close

    for filename in l:
        f = pf.open(filename)
        propid = f[0].header['PROPOSID']
        f.close()

        os.system('ls ../data/persist/%d/*persist.fits > foo.txt' % propid)
        f = open('foo.txt')
        persists = f.read().split('_persist.fits\n')
        persists = ''.join(persists).split('../data/persist/%d/' % propid)
        f.close

        if filename.split('/')[-1][:-10] in persists:
            pass
        else:
            print filename.split('/')[-1][:-10], propid

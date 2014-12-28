from random import *

for i in range(64):
    for j in range(2):
        print('C(0x%016x, 0x%016x)' % (getrandbits(64),getrandbits(64)), end=', ')
    print('')


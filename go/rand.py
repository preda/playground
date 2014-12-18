from random import *

for i in range(16):
    for j in range(4):
        print('0x%016xull' % (getrandbits(64),), end=', ')
    print('')


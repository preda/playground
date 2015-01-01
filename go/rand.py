from random import *

n = 64
count = [0] * n
count[0] = 1

for i in range(36 * 2):
    r = getrandbits(6)
    c2 = count[:]
    for i in range(n): c2[r ^ i] += count[i]
    count = c2
    print(r, count, '\n')

        #print('C(0x%016x, 0x%016x)' % (getrandbits(64),getrandbits(64)), end=', ')

import pyprimes

#f = 100.
#for p in pyprimes.primes(11, 10000000):
#    f *= (p - 1) / float(p)
#print(f)

n = 0
#1047139
#1047157
for p in pyprimes.primes(13, 512 * 1024):
    print("%d, " % p, end='')
    n += 1
    if n >= 1024:
        print()
        n = 0

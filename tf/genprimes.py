import pyprimes

if False:
    f = 100.
    for p in pyprimes.primes(500000, 1000000):
        f *= (p - 1) / float(p)
    print(f)

if True:
    n = 0
    for p in list(pyprimes.nprimes(23 * 1024 + 5))[5:]:
        print("%7d, " % p, end='')
        n += 1
        if n >= 8:
            print()
            n = 0

#for p in pyprimes.primes(13, 2 * 1024 * 1024):
#    print("%d, " % p, end='')
#    n += 1
#    if n >= 256:
#        print()
#        n = 0

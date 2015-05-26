import pyprimes

n = 0
for p in pyprimes.primes(67, 64 * 1024):
    print("%5d, " % p, end='')
    n += 1
    if n >= 32:
        print()
        n = 0

import pyprimes

f = 100.
for p in pyprimes.primes(11, 10000000):
    f *= (p - 1) / float(p)
print(f)

#n = 0
#for p in pyprimes.primes(67, 64 * 1024):
#    print("%5d, " % p, end='')
#    n += 1
#    if n >= 32:
#        print()
#        n = 0

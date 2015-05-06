print "// Do not edit: generated by gensieve.py\ntypedef unsigned long long u64;"

def gen(p, op):
    print "void sieve%02d(u64 *p, u64 *end) { do {" % p
    bit = 0
    for i in range(p):
        mask = 0
        while (bit < 64):
            mask |= (1 << bit)
            bit += p
        bit -= 64
        print "p[%d] %s 0x%016x;" % (i, op, mask)
    print "p += %d;" % p
    print "} while (p < end); }"


gen(3, '=')
primes5 = (5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61) 
for p in primes5:
    gen(p, '|=')

print "void sieve(int n, u64 *p, u64 *end) {\nswitch (n) {"
for p in (3,) + primes5:
    print "case %02d: sieve%02d(p, end); break;" % (p, p)
print "}}"

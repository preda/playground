from math import log

#(p, k) = (800000479, 3089082494924)
#(p, k) = (60008807, 259574105937)
(p, k) = (3321931973, 4016423448)

m = 2 * k * p + 1
print "m %x p %x" % (m, p)

print '%x' % ((1 << (28 + 96)) % m) 

def clz(x):
    c = 0
    while ((x & 0x80000000) == 0):
        c += 1
        x <<= 1
    return c

print clz(p)

def mod(x, rawM):
    print "%x %x"%(x, rawM)
    shift = clz(rawM >> 64) - 2
    m = rawM << shift
    R = ((1 << 64) - 1) / ((m >> 61) + 1)
    x <<= shift
    n = (x >> 128) * R >> 32
    print "e %d R %d n %d" % ((x>>128), R, n)
    x = x - ((m * n) << 35)
    print "x %x" % x
    n = ((x >> 100) * R) >> 32
    x = x - ((m * n) << 7)
    print "x %x" % x
    n = ((x >> 72) * R) >> (32 + 21)
    print("mn %x" % (m * n));
    x = x - m * n
    print "x %x" % x


mod(1 << (28 + 96), m)
    
bit = 31 - clz(p)
a = 1
while bit >= 0:
    on = p & (1<<bit)
    print "bit %d %d" % (bit, on)
    a = (a * a) % m
    if on: a = (a << 1) % m

    b = (a << 96) % m
    
    print '%x - %x'% (a, b)
    bit -= 1
print a




def mprime(m):
    m = (m >> 1) + 1
    u = m
    v = 0x80000000;
    for i in range(31):
        odd = u & 1
        u >>= 1
        v >>= 1
        if (odd):
            u += m
            v |= 0x80000000;
    return v

def reduct(x, m, mp):
    return (x + ((x * mp) & 0xffffffff) * m) >> 32

n = 2903480177832398150183

#print "0x%08x, 0x%08x, 0x%08x" % (n >> 64, (n >> 32) & 0xffffffff, n & 0xffffffff) 

#for m in (41, 0xff, 0xffffffff, 1023, 955):
#    mp = mprime(m)
#    print "%x %x %x" % (m, mp, m * mp)

#    for a in (1, 2, 24, 25, 10342):
#        a = a % m
#        aa = (a << 32) % m
#        a2a = reduct(aa * aa, m, mp)
#        r = reduct((aa + m) * (aa + m), m, mp)
        #print a, reduct(aa, m, mp), (a*a) % m, reduct(a2a, m, mp), a2a % m, r % m



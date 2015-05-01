from math import log

k = 3089082494924
p = 800000479
a = 1
m = 2 * k * p + 1
print "m %x p %x" % (m, p)
bit = int(log(p, 2))
U = (1 << 96) % m

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



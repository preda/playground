def add(a, b): return list(x+y for (x, y) in zip(a, b))
def neg(a): return list(-x for x in a)
def sub(a, b): return add(a, neg(b))
def sh(a, e):
    if e < 0: return sh(neg(a), len(a) + e)
    assert e < len(a)
    return neg(a[-e:]) + list(a[:-e])

def posh(a, e): return a[-e:] + a[:-e]
    
def ah(x, y):
    assert (x & 1) == (y & 1)
    return (x >> 1) + (y >> 1) + (x & 1)

def addh(a, b): return list(ah(x, y) for (x, y) in zip(a, b))
def subh(a, b): return addh(a, neg(b))

# 3 nc's
def poly2(v, nc, posh):
    (x, y) = (v[::2], v[1::2])
    (a, c) = (nc(x), nc(y))
    b = sub(add(a, c), nc(sub(x, y)))
    return tuple(x for l in zip(add(a, posh(c, 1)), b) for x in l)

# 8 nc's
def poly4(v, nc, posh):
    #a b c d e f g h
    (a, b, c, d) = zip(*(v[4 * i: 4 * i + 4] for i in range(4)))
    (e, f, g, h) = (a, sh(b, 1), sh(c, 2), sh(d, 3))

    (a, b, c, d, e, f, g, h) = (add(a,c), add(b,d), sub(a,c), sh(sub(b,d), 2), add(e, g), add(f, h), sub(e, g), sh(sub(f, h), 2))
    (a, b, c, d, e, f, g, h) = (add(a, b), sub(a, b), add(c, d), sub(c, d), add(e, f), sub(e, f), add(g, h), sub(g, h))

    (a, b, c, d, e, f, g, h) = map(nc, (a, b, c, d, e, f, g, h))

    (a, b, c, d, e, f, g, h) = (addh(a, b), subh(a, b), addh(c, d), subh(c, d), addh(e, f), subh(e, f), addh(g, h), subh(g, h))
    (d, h) = (sh(d, -2), sh(h, -2))
    (a, b, c, d, e, f, g, h) = (addh(a, c), addh(b, d), subh(a, c), subh(b, d), addh(e, g), addh(f, h), subh(e, g), subh(f, h))
    (f, g, h) = (sh(f, -1), sh(g, -2), sh(h, -3))
    if d != h: print('*', d, h)
    assert d == h
    (a, b, c, e, f, g) = (addh(a, e), addh(b, f), addh(c, g), subh(a, e), subh(b, f), subh(c, g))
    
    (e, f, g) = (posh(e, 1), posh(f, 1), posh(g, 1))
    return list(x for l in zip(add(a, e), add(b, f), add(c, g), d) for x in l)

# 2 muls
def nc2(v):
    (a, b) = v
    return ((a + b) * (a - b), a*b*2)

# 2 muls
def c2(v):
    (a, b) = v
    c = a * b * 2
    return ((a - b)**2 + c, c)

def c4(v): return poly2(v, c2, posh)
def c4slow(v):
    (a, b, c, d) = v
    return (a*a + c*c + 2*b*d, 2*(a*b + c*d), 2*a*c + b*b + d*d, 2*(a*d + b*c))

# 6 muls
def nc4(v):
    return poly2(v, nc2, sh)

# negacyclic auto conv of 4 in 10 muls
def nc4slow(v):
    (a, b, c, d) = v
    return (a*a - c*c - 2*b*d, 2*(a*b - c*d), 2*a*c + b*b - d*d, 2*(a*d + b*c))

v = (2, -1, 3, 1)
print(c4(v), c4slow(v))

# negacyclic auto conv of 4 in 7 muls
def nc4a(v):
    (a, b, c, d) = v
    (apc, amc) = (a + c, a - c)
    (bpd, bmd) = (b + d, b - d)
    x = apc * bmd
    y = amc * bpd
    return (apc * amc - 2*b*d, x + y, bpd * bmd + 2*a*c, y - x + 4*b*c)
    
def nc16(v):
    return poly4(v, nc4, sh)

def c16(v): return poly4(v, c4, posh)

def cycl(v):
    (a, b, c, d) = zip(*(v[4 * i: 4 * i + 4] for i in range(4)))
    (a, b, c, d) = (add(a, c), add(b, d), sub(a, c), sh(sub(b, d), 2))
    (a, b, c, d) = (add(a, b), sub(a, b), add(c, d), sub(c, d))
    (a, b, c, d) = map(nc4, (a, b, c, d))
    (a, b, c, d) = (addh(a, b), subh(a, b), addh(c, d), subh(c, d))
    d = sh(d, -2)
    (a, b, c, d) = (addh(a, c), addh(b, d), subh(a, c), subh(b, d))
    return (a, b, c, d)

def negcycl(v):
    (a, b, c, d) = zip(*(v[4 * i: 4 * i + 4] for i in range(4)))
    (b, c, d) = (sh(b, 1), sh(c, 2), sh(d, 3))    
    (a, b, c, d) = (add(a, c), add(b, d), sub(a, c), sh(sub(b, d), 2))
    (a, b, c, d) = (add(a, b), sub(a, b), add(c, d), sub(c, d))
    (a, b, c, d) = map(nc4, (a, b, c, d))
    (a, b, c, d) = (addh(a, b), subh(a, b), addh(c, d), subh(c, d))
    d = sh(d, -2)
    (a, b, c, d) = (addh(a, c), addh(b, d), subh(a, c), subh(b, d))    
    return (a, sh(b, -1), sh(c, -2), sh(d, -3))

def negcycl_b(v):
    (a, b, c, d) = zip(*(v[4 * i: 4 * i + 4] for i in range(4)))
    (b, c, d) = (sh(b, -1), sh(c, -2), sh(d, -3))    
    (a, b, c, d) = (add(a, c), add(b, d), sub(a, c), sh(sub(b, d), 2))
    (a, b, c, d) = (add(a, b), sub(a, b), add(c, d), sub(c, d))
    (a, b, c, d) = map(nc4, (a, b, c, d))
    (a, b, c, d) = (addh(a, b), subh(a, b), addh(c, d), subh(c, d))
    (a, b, c) = (addh(a, c), addh(b, sh(d, -2)), subh(a, c))
    return (a, sh(b, 1), sh(c, 2))

def nc16fancy(v):
    (a, b, c, d) = cycl(v)
    (e, f, g) = negcycl_b(v)
#    (a, b, c, d, e, f, g, h) = (addh(a, e), addh(b, f), addh(c, g), addh(d, h),
#                                subh(a, e), subh(b, f), subh(c, g), subh(d, h))
    (a, b, c, e, f, g) = (addh(a, e), addh(b, f), addh(c, g), subh(a, e), subh(b, f), subh(c, g))
    return tuple(x for l in zip(add(a, sh(e, 1)), add(b, sh(f, 1)), add(c, sh(g, 1)), d) for x in l)
#return (a, b, c, d, e, f, g, h)

def nc16slow(v):
    out = [0] * 16
    for i in range(16):
        for j in range(16):
            p = v[i] * v[j]
            if i + j >= 16: p = -p
            out[(i + j) & 15] += p
    return out

def c16slow(v):
    out = [0] * 16
    for i in range(16):
        for j in range(16):
            out[(i + j) & 15] += v[i] * v[j]
    return out



v = [] + list(range(16))
v[0] = -12
print(nc16(v))
print(nc16slow(v))
print(nc16fancy(v))

#print(c16(v), c16slow(v))

def nc8(v):
    return poly2(v, nc4, sh)



def slow_nc8(v):
    out = [0] * 8
    for i in range(8):
        for j in range(8):
            p = v[i] * v[j]
            if i + j > 7: p = -p
            out[(i + j) & 7] += p
    return out

def nc8d(v):
    (a, b, c, d, e, f, g, h) = v
    return (
        a*a - e*e - 2 * (b*h + c*g + d*f),
        2 * (a*b - c*h - d*g - e*f),
        b*b - f*f + 2 * (a*c - d*h - e*g),
        2 * (a*d + b*c - e*h - f*g),
        c*c - g*g + 2 * (a*e + b*d - f*h),
        2 * (a*f + b*e + c*d - g*h),
        d*d - h*h + 2 * (a*g + b*f + c*e),
        2 * (a*h + b*g + c*f + d*e))

v = [0] * 8
v[0] = 5
v[1] = 2
v[2] = 3
v[3] = -1
v[4] = -2
v[5] = 4
v[6] = -3
v[7] = 1

print(slow_nc8(v), nc8(v), nc8d(v))
print nc8(v) == nc8d(v)

v = (2, 3, 5, 7)
#print(nc4(v), nc4a(v))

exit(0)



            








def negasq(v):
    (a, b) = v
    return (a*a - b*b, 2*a*b)

def add(a, b):
    return (a[0] + b[0], a[1] + b[1])

def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])

def shift(a):
    return (-a[1], a[0])

def rshift(a):
    return (a[1], -a[0])

def negcore(v, add, sub, shift, rshift, sq=None):
    v = dif(v, add, sub, shift)
    print('bef ', v)
    v = tuple((sq or negasq)(x) for x in v)
    print('aft ', v)
    v = dit(v, add, sub, rshift)
    return v

def negconv(v):
    (a, b, c, d) = v
    v = ((a, c), (b, d), (0, 0), (0, 0))
    v = negcore(v, add, sub, shift, rshift)
    print('x ', v)
    ((a, e), (b, f), (c, g), (d, h)) = v
    v = ((a - g)/4, (b - h)/4, (c + e)/4, (d + f)/4)
    return v
    
def dif(v, add, sub, shift):
    (a, b, c, d) = v
    (a, b, c, d) = (add(a, c), add(b, d), sub(a, c), shift(sub(b, d)))
    (a, b, c, d) = (add(a, b), sub(a, b), add(c, d), sub(c, d))
    return (a, b, c, d)

def dit(v, add, sub, shift):
    (a, b, c, d) = v
    (a, b, c, d) = (add(a, b), sub(a, b), add(c, d), sub(c, d))
    sd = shift(d)
    (a, b, c, d) = (add(a, c), add(b, sd), sub(a, c), sub(b, sd))
    return (a, b, c, d)

def fft(v, add, sub, shift, rshift):
    return dit(dif(v, add, sub, shift), add, sub, rshift)

def conv(v):
    (a, b, c, d) = dif(v, add1, sub1, shift1)
    v = (a*a, b*b, c*c, d*d)
    return dit(v, add1, sub1, rshift1)

def add1(a, b): return a + b
def sub1(a, b): return a - b
def shift1(a): return a * 1j
def rshift1(a): return - a * 1j
def sq1(a): return a * a
v = (2+1j, 3, 7, 4j)
v = (1, 5, 0, 0)
v = (3, 1, 0, 0)
print(fft(v, add1, sub1, shift1, rshift1))
print(conv(v))
print(negcore(v, add1, sub1, shift1, rshift1, sq1))

v = ((1, 0), (0, 0), (0, 0), (0, 0))
print(fft(v, add, sub, shift, rshift))

def gramnegconv(v):
    (a, b, c, d) = v
    return (a * a - c * c - 2 * b * d, 2 * a * b - 2 * c * d, 2 * a * c + b * b - d * d, 2 * a * d + 2 * b * c)

#print(conv(v, add, sub, shift, rshift))
v = (1, 5, 2, 3)
#v = (1, 5, 0, 0)
#v = (1, 0, 1, 10)
#v = (1, 0, 0, 2)
print(negcore(v, add1, sub1, shift1, rshift1, sq1))
print(negconv(v))
print(gramnegconv(v))

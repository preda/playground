
def nc4(v):
    (a, b, c, d) = v
    return (a*a - c*c - 2*b*d, 2*(a*b - c*d), 2*a*c + b*b - d*d, 2*(a*d + b*c))

def p4(x0, x1):
    (a, b, c, d) = x0
    (e, f, g, h) = x1
    return (a+e, b+f, c+g, d+h)

def m4(x0, x1):
    (a, b, c, d) = x0
    (e, f, g, h) = x1
    return (a-e, b-f, c-g, d-h)

def rot(v):
    (a, b, c, d) = v
    return (-d, a, b, c)

def nc8(v):
    (a, b, c, d, e, f, g, h) = v
    x0 = (a + b, c + d, e + f, g + h)
    x1 = (a - f, c - h, e + b, g + d)
    x2 = (a - b, c - d, e - f, g - h)
    x3 = (a + f, c + h, e - b, g - d)
    (x0, x1, x2, x3) = map(nc4, (x0, x1, x2, x3))
    (x0, x2) = (p4(x0, x2), m4(x0, x2))
    #(x1, x3) = (p4(x1, x3), m4(x1, x3))
    x1 = p4(x1, x3)
    (x0, x1) = (p4(x0, x1), m4(x0, x1))
    #(a, b, c, d) = x3
    #x3 = (-c, -d, a, b)
    #x2 = m4(x2, x3)
    x2 = map(lambda x: 2*x, x2)
    x0 = p4(x0, rot(x1))
    #(x2, x3) = (m4(x2, x3), p4(x2, x3))
    #print('* ', x3)
    # x2 = p4(x2, rot(x3))
    (a, c, e, g) = x0
    (b, d, f, h) = x2
    return map(lambda x: x/4, (a, b, c, d, e, f, g, h))

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

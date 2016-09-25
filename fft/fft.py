#!/usr/bin/python3

import numpy

fft = numpy.fft.fft

a = [0] * 2048
a[0] = 1+2j
a[1] = 3
f = fft(a)
#for i in range(2048):
#    print(f[i], end='')
#    if (i & 3)==3: print('')
#exit(0)

from math import *


for r in range(17):
    x = cos(pi * r / 32)
    print("%f" % x)
#exit(0)

l = set()
for r in range(8):
    for c in range(8):
        p = r * c
        if (p >= 32): p -= 32
        if (p >= 16): p -= 16
        if (p >= 8): p = 16 - p
        l.add(p)
print(len(l))
print(l)
        #print(e**(-1j * r * c * pi / 32),)


def fft8(x):
    for i in range(4):
        (x[i], x[i+4]) = (x[i]+x[i+4], x[i]-x[i+4])
    x[5] *= e**(-1j * pi / 4)
    x[6] *= e**(-1j * pi / 2)
    x[7] *= e**(-3j * pi / 4)
    for i in range(2):
        (x[i], x[i+2]) = (x[i] + x[i+2], x[i] - x[i+2])
        (x[i+4], x[i+6]) = (x[i+4] + x[i+6], x[i+4] - x[i+6])
    x[3] *= e**(-1j * pi / 2)
    x[7] *= e**(-1j * pi / 2)
    for i in range(4):
        (x[i*2], x[i*2+1]) = (x[i*2]+x[i*2+1], x[i*2] - x[i*2+1])
    return x
        
def evenOdd(v):
    return zip(*[(v[2*i],v[2*i+1]) for i in range(len(v)//2)])

def revbin(v):
    if len(v) <= 2:
        return v
    (even, odd) = evenOdd(v)
    return revbin(even) + revbin(odd)

print(revbin(range(32)))

print(revbin(range(16)))

print(revbin(fft8([1+2j, 3, 0, 0, 0, 0, 0, 0])))

bitpos = revbin(range(8))
#print(bitpos)
#print(revbin(range(16)))

def makeCoef(sign, n):
    coef = []
    for k in range(n):
        arg = sign * pi * k / n
        q = e ** (arg * 1j)   #cos(arg) + sin(arg) * 1j
        coef.append(q)
    bitpos = revbin(range(n))
    return [coef[bitpos[i]] for i in range(n)]
        
coef = makeCoef(-1, 4)
#print(coef)
invCoef = makeCoef(1, 4)

def step(round, vi, vo, idx, coef):
    s = 2**round
    block = idx // s
    k = idx % s
    i = block * s * 2 + k
    a = vi[i]
    b = vi[i + s]
    c = coef[k]
    #if k == 1 and round == 2: print('***', c)
    cb = c * b
    vo[idx * 2] = a + cb
    vo[idx * 2 + 1] = a - cb

def fftx(vi, p, coef):
    n = 2**p
    vo = [0] * len(vi)
    (vi, vo) = (vo, vi)
    for round in range(p):
        (vi, vo) = (vo, vi)
        for idx in range(n // 2):
            step(round, vi, vo, idx, coef)
        #print(round, vo)
    return vo

def fft(vi):
    return fftx(vi, 3, coef)

def ffti(vi):
    return [x/8 for x in fftx(vi, 3, invCoef)]

vi = [1, 2, 3, 4, 5, 6, 7, 8]
vi = [0, 0, 0, 0, 0, 0, 0, 1]
vi = [0, 0, 1, 0, 0, 0, 0, 0]
vi = [1, 1, -1, -1, -1j, -1j, 1j, 1j]

vi = [1+2j, 3+4j, 5+6j, -2-3j, -4-5j, 10, 20, 3j]

vo = fft(vi)
print(vo)

vo = ffti(vo)
print(vo)

def mul(a, b):
    return [x*y for (x, y) in zip(a, b)]



#a = [2, 0, 0, 0, 0, 0, 0, 0] #
#b = [1, 0, 0, 0, 1, 0, 0, 0] # [0, 0, 0, 0, 2, 0, 1, 1]
print(revbin([0, 1, 2, 3, 4, 5, 6, 7]))
a = list(revbin([1, 2, 3, 4, 0, 0, 0, 0]))
b = list(revbin([2, 0, 1, 1, 0, 0, 0, 0]))
#print(mul(a, b))
fa = fft(a)
fb = fft(b)
fab = mul(fa, fb)
#print('\n')
#print(fa)
#print(fb)
#print(fab)
ab = ffti(fab)
#print(revbin(ab))

from math import *

mc = [0] * 40
mc[0] = 1
mc[1] = 2
mc[2] = 6

for i in range(3, 40):
    m = 3 * mc[i - 1]
    for j in range((i + 1) // 2, i - 1):
        m = min(m, mc[j] * (1 << (i - j + 1)))
    #if i >= 4: m = min(m, mc[i - 2] * 7)
    mc[i] = m

for i in range(1, 26):
    print("%10d %10d %5.1f  %d" % (1<< i, mc[i], mc[i] / (1 << i), i))  

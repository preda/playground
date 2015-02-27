for i in range(2**9):
    v = 0
    for p in range(9):
        if i & (1 << p):
            v += 3 ** p
    if i % 16 == 0: print
    print '%4d,' % v,

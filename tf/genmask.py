for p in (3, 5, 7, 11, 13, 17, 19, 23, 29, 31): #, 37, 41, 43, 47, 53, 59, 61):
    mask = 0
    i = 0
    while (i < 32):
        mask |= 1 << i
        i += p
    print '%2d 0x%08x' % (p, mask)

    

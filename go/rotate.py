def rotate(x):
    y = 0
    for p in range(6):
        if x & (1<<p): y += 1<<(5-p)
    return y
        
for i in range(64):
    print "%2d,"%rotate(i),

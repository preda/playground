typedef unsigned char byte;

enum {
  SIZE_Y = 6,
  SIZE_X = 6,

  BIG_X = SIZE_X + 2,
  BIG_Y = SIZE_Y + 2,
  
  N = SIZE_X * SIZE_Y,
  BIG_N = BIG_X * BIG_Y,

  BLACK = 0,
  WHITE = 1,
  EMPTY = 2,
  BROWN = 3,

  MAX_GROUPS = N / 4,
};

#pragma once

typedef unsigned char byte;
typedef unsigned long long uint64_t;

enum {
  SIZE_Y = 6,
  SIZE_X = 6,

  X_SHIFT = 3,
  BIG_X = (1 << X_SHIFT),
  BIG_Y = SIZE_Y + 2,
  DELTA = BIG_X,
  
  N = SIZE_X * SIZE_Y,
  BIG_N = BIG_X * BIG_Y,

  BLACK = 0,
  WHITE = 1,
  EMPTY = 2,
  BROWN = 3,

  MAX_GROUPS = N / 2,
};

static inline int P(int y, int x) { return ((y + 1) << X_SHIFT) + x; }
static inline int Y(int pos) { return (pos >> X_SHIFT) - 1; }
static inline int X(int pos) { return pos & (BIG_X - 1); }

static bool isBlackOrWhite(int color) { return color == BLACK || color == WHITE; }


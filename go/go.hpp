#pragma once

typedef unsigned char byte;
typedef unsigned long long uint64_t;
typedef unsigned __int128 uint128_t;

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
  TOTAL_POINTS = SIZE_X * SIZE_Y,
};

static inline int P(int y, int x) { return ((y + 1) << X_SHIFT) + x; }
static inline int Y(int pos) { return (pos >> X_SHIFT) - 1; }
static inline int X(int pos) { return pos & (BIG_X - 1); }

static inline bool isBlackOrWhite(int color) { return color == BLACK || color == WHITE; }

/*
#define LINE(i) (((1ull << SIZE_X) - 1) << (i * BIG_X))
#if SIZE_Y == 6
constexpr uint64_t INSIDE = LINE(1) | LINE(2) | LINE(3) | LINE(4) | LINE(5) | LINE(6);
#elif SIZE_Y == 5
constexpr uint64_t INSIDE = LINE(1) | LINE(2) | LINE(3) | LINE(4) | LINE(5);
#else
#error foo
#endif
#undef LINE

static constexpr uint64_t insideBorder() {
  uint64_t inside = 0;
  for (int y = 0; y < SIZE_Y; ++y) {
    for (int x = 0; x < SIZE_X; ++x) {
      inside |= (1 << P(y, x));
    }
  }
  return inside;
}
*/

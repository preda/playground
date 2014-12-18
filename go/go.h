#pragma once

typedef unsigned char byte;
typedef unsigned long long uint64_t;

enum {
  SIZE_Y = 6,
  SIZE_X = 6,

  BIG_X = SIZE_X + 2,
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

static int pos(int y, int x) { return (y + 1) * BIG_X + x + 1; }
static bool isBlackOrWhite(int color) { return color == BLACK || color == WHITE; }
static bool isValid(int y, int x) { return y >= 0 && y < SIZE_Y && x >= 0 && x < SIZE_X; }

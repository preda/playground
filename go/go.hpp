// (c) Copyright 2014 Mihai Preda. All rights reserved.

#pragma once

#include <stdint.h>

typedef unsigned char byte;
typedef signed char sbyte;
typedef unsigned __int128 uint128_t;

#define SIZE_X 4
#define SIZE_Y 4

enum {
  X_SHIFT = 3,
  BIG_X = (1 << X_SHIFT),
  BIG_Y = SIZE_Y + 2,
  DELTA = BIG_X,
  
  N = SIZE_X * SIZE_Y,
  BIG_N = BIG_X * BIG_Y,
  MAX_GROUPS = N,
  PASS = 1,
};

static inline int P(int y, int x) { return ((y + 1) << X_SHIFT) + x; }
static inline int Y(int pos) { return (pos >> X_SHIFT) - 1; }
static inline int X(int pos) { return pos & (BIG_X - 1); }

// constexpr bool isBlackOrWhite(int color) { return color == BLACK || color == WHITE; }

#define SET(p, bits) bits |= (1ull << (p))

// Consts generated by constexpr.cpp
//6x6
#if SIZE_X == 6 and SIZE_Y == 6
#define BORDER 0xff40c0c0c0c0c0ff
#define INSIDE 0x3f3f3f3f3f3f00
#define HALF_Y 0x3f3f3f00
#define HALF_X 0x7070707070700
#define HALF_DIAG 0x2030383c3e3f00
#endif

//5x5
#if SIZE_X == 5 and SIZE_Y == 5
#define BORDER 0xff20a0a0a0a0ff
#define INSIDE 0x1f1f1f1f1f00
#define HALF_Y 0x1f1f1f00
#define HALF_X 0x70707070700
#define HALF_DIAG 0x10181c1e1f00
#endif

//4x4
#if SIZE_X == 4 and SIZE_Y == 4
#define BORDER 0xff10909090ff
#define INSIDE 0xf0f0f0f00
#define HALF_Y 0xf0f00
#define HALF_X 0x303030300
#define HALF_DIAG 0x80c0e0f00
#endif

//3x3
#if SIZE_X == 3 and SIZE_Y == 3
#define BORDER 0xff088888ff
#define INSIDE 0x7070700
#define HALF_Y 0x70700
#define HALF_X 0x3030300
#define HALF_DIAG 0x4060700
#endif

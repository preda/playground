// (c) Copyright 2014 Mihai Preda. All rights reserved.

#pragma once

#include <stdint.h>

#define PURE const __attribute__((warn_unused_result))

typedef unsigned char byte;
typedef signed char sbyte;
typedef unsigned __int128 uint128_t;
typedef uint64_t u64;


#define SQ_SIZE 3
#define SIZE_X SQ_SIZE
#define SIZE_Y SQ_SIZE

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

#define SET(p, bits) bits |= (1ull << (p))

// Consts generated by constexpr.cpp
//6x6
#if SIZE_X == 6 and SIZE_Y == 6
#define BORDER 0xff40c0c0c0c0c0ff
#define INSIDE 0x3f3f3f3f3f3f00
#endif

//5x5
#if SIZE_X == 5 and SIZE_Y == 5
#define BORDER 0xff20a0a0a0a0ff
#define INSIDE 0x1f1f1f1f1f00
#endif

//4x4
#if SIZE_X == 4 and SIZE_Y == 4
#define BORDER 0xff10909090ff
#define INSIDE 0xf0f0f0f00
#endif

//3x3
#if SIZE_X == 3 and SIZE_Y == 3
#define BORDER 0xff088888ff
#define INSIDE 0x7070700
#endif


inline int size(uint64_t bits) { return __builtin_popcountll(bits); }

inline bool IS(int p, auto bits) { return bits & (1LL << p); }

inline void CLEAR(int p, u64 &bits) { bits &= ~(1LL << p); }
// #define CLEAR(p, bits) bits &= ~(1LL << p)

inline int POP(u64 &bits) { int r = firstOf(bits); CLEAR(r, bits); return r; }

inline int firstOf(u64 bits) { return __builtin_ctzll(bits); }
inline int firstOf(u32 bits) { return __builtin_ctz(bits);   }

template<typename T>
class Bits {
  struct BitsIt {
    T bits;
    int operator*() { return firstOf(bits); }
    void operator++() { bits &= bits - 1; }
    bool operator!=(BitsIt o) { return bits != o.bits; }
  };

  T bits;
public:
  Bits(T bits) : bits(bits) {}
  BitsIt begin() { return {bits}; }
  BitsIt end() { return {0}; }
};

template<typename T, int N>
class vect {
  T v[N];
  int _size = 0;

public:
  void push(T t) { assert(_size < N); v[_size++] = t; }
  T pop()        { assert(_size > 0); return v[--_size]; }
  int size() { return _size; }
  bool isEmpty() { return _size <= 0; }
  bool has(T t) {
    for (T e : *this) { if (e == t) { return true; } }
    return false;
  }
  void clear() { _size = 0; }
  
  T *begin() { return v; }
  T *end() { return v + _size; }
  T operator[](int i) { return v[i]; }
};

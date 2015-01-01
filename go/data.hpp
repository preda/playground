#pragma once

#include <stdint.h>

inline int firstOf(uint64_t bits) { return __builtin_ctzll(bits); }

class Bits {
  struct BitsIt {
    uint64_t bits;
    int operator*() { return firstOf(bits); }
    void operator++() { bits &= bits - 1; }
    bool operator!=(BitsIt o) { return bits != o.bits; }
  };

  uint64_t bits;
public:
  Bits(uint64_t bits) : bits(bits) {}
  BitsIt begin() { return {bits}; }
  BitsIt end() { return {0}; }
};

template<typename T, int N>
class Vect {
  T v[N];
  int size = 0;

public:
  void push(T t) { v[size++] = t; }
  T pop() { return v[--size]; }
  bool isEmpty() { return size <= 0; }
  bool has(T t) {
    for (T e : *this) { if (e == t) { return true; } }
    return false;
  }
  void clear() { size = 0; }
  
  T *begin() { return v; }
  T *end() { return v + size; }
};

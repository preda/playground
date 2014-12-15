// -*- C++ -*-

#include "go.h"

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

class Bitset {
  unsigned long long bits = 0;
public:
  bool testAndSet(int p) {
    unsigned long long mask = (1ull << p);
    bool bit = bits & mask;
    if (!bit) { bits |= mask; }
    return bit;
  }

  bool test(int p) {
    return bits & (1ull << p);
  }
};

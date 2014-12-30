#pragma once

typedef unsigned long long uint64_t;

class Bits {
  struct BitsIt {
    uint64_t bits;
    int operator*() { return __builtin_ctzll(bits); }
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

class Bitset {
  unsigned long long bits = 0;
public:

  bool operator[](int p) const { return bits & (1ull << p); }
  
  bool testAndSet(int p) {
    unsigned long long mask = (1ull << p);
    bool bit = bits & mask;
    if (!bit) { bits |= mask; }
    return bit;
  }

  void set(int p) {
    bits |= (1ull << p);
  }

  void clear() {
    bits = 0;
  }

  void operator|=(Bitset o) {
    bits |= o.bits;
  }

  int size() const {
    return __builtin_popcountll(bits);
  }
};

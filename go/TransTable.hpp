#pragma once

#include "go.hpp"

#define LOCK_BITS 48

struct Slot {
  uint64_t lock:LOCK_BITS;
  signed char min;
  signed char max;

  bool isEmpty() { return lock == 0 && min == 0 && max == 0; }
  ScoreBounds score() { return {min, max}; }
};

class TransTable {
private:
  Slot *slots;

public:
  TransTable();
  ~TransTable();

  ScoreBounds lookup(uint128_t hash);
  void set(uint128_t hash, int min, int max);
};

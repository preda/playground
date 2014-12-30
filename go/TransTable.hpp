#pragma once

#include "go.hpp"

#define LOCK_BITS 48

struct NodeInfo {
  signed char min;
  signed char max;
};

struct Slot {
  uint64_t lock:LOCK_BITS;
  signed char min;
  signed char max;

  bool isEmpty() { return lock == 0 && min == 0 && max == 0; }
  NodeInfo info() { return {min, max}; }
};

class TransTable {
private:
  Slot *slots;
  // NodeInfo lookup(uint64_t pos, uint64_t lock);  
  // void set(uint64_t pos, uint64_t lock, NodeInfo info);

public:
  TransTable();
  ~TransTable();

  NodeInfo lookup(uint128_t hash);
  void set(uint128_t hash, int min, int max);
};

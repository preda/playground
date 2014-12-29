#pragma once

#include "go.hpp"

#define LOCK_BITS 42

struct SlotInfo {
  byte pos;
  signed char min;
  signed char max;
};

struct Slot {
  uint64_t lock:LOCK_BITS;
  byte pos:6;
  signed char min;
  signed char max;

  bool isEmpty() {
    return pos == 0;
    /*
    union {
      Slot s;
      uint64_t v;
    } u{*this};
    return u.v == 0;
    */
  }
};

/*
struct HashKey {
  uint64_t pos;
  uint64_t lock;
};
*/

class TransTable {
private:
  Slot *slots;
  SlotInfo lookup(uint64_t pos, uint64_t lock);  
  void set(uint64_t pos, uint64_t lock, SlotInfo info);

public:
  TransTable();
  ~TransTable();

  SlotInfo lookup(uint128_t hash);
};

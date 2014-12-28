#pragma once

#include "go.hpp"

struct SlotInfo {
  signed char score;
  byte pos;
  bool over, under;
};



struct Slot {
  uint64_t lock:48;
  signed char score;
  byte pos:6;
  bool over:1;
  bool under:1;

  bool isEmpty() {
    union {
      Slot s;
      uint64_t v;
    } u{*this};
    return u.v == 0;
  }
};

struct HashKey {
  uint64_t pos;
  uint64_t lock;
};

HashKey makeKey(uint128_t hash);

class TransTable {
private:
  Slot *slots;

public:
  TransTable();
  ~TransTable();
  
  SlotInfo lookup(uint64_t pos, uint64_t lock);  
  void set(uint64_t pos, uint64_t lock, SlotInfo info);
};

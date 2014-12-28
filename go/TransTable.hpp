#pragma once

#include "go.hpp"

struct Info {
  byte level;
  byte value;
};

struct Slot {
  uint64_t lock:48;
  byte level;
  byte value;
};

struct HashKey {
  uint64_t pos;
  uint64_t lock;
};

HashKey makeKey(uint128_t hash);

class TransTable {
private:
  int reserveBits;  
  uint64_t size;
  uint64_t mask;
  Slot *slots;

public:
  TransTable();
  ~TransTable();
  
  Slot *lookup(uint64_t pos, uint64_t lock);  
  void set(uint64_t pos, uint64_t lock, byte level, byte value);
};

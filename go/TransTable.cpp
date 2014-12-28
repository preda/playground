#include "TransTable.hpp"
#include <stdio.h>

#define SLOT_BITS 30
#define RES_BITS 4
#define N 8

constexpr uint64_t SIZE = (1ull << SLOT_BITS) - (1ull << (SLOT_BITS - RES_BITS));
constexpr uint64_t MASK = (1ull << SLOT_BITS) - 1;

TransTable::TransTable() :
  slots(new Slot[SIZE + N - 1])
{
  printf("Size %.2f GB, slot size %ld\n", SIZE * sizeof(Slot) / (1024 * 1024 * 1024.0f), sizeof(Slot));
}

TransTable::~TransTable() {
  delete[] slots;
  slots = 0;
}

HashKey makeKey(uint128_t hash) {
  uint64_t lock = (hash >> 64) & ((1ull << 48) - 1);
  uint64_t pos  = hash & MASK;
  while (pos >= SIZE) {
    pos >>= RES_BITS;
  }
  return {pos, lock};
}

Slot *TransTable::lookup(uint64_t pos, uint64_t lock) {
  for (Slot *s = slots + pos, *end = s + N; s < end; ++s) {
    if (s->lock == lock) {
      return s;
    }
  }
  return 0;
} 

void TransTable::set(uint64_t pos, uint64_t lock, byte level, byte value) {
  int minLevel = 1000;
  Slot *minSlot = 0;
  for (Slot *s = slots + pos, *end = s + N; s < end; ++s) {
    int sLevel = s->level;
    if (sLevel == 0) {
      *s = {lock, level, value};
      return;
    } else if (sLevel < minLevel) {
      minLevel = sLevel;
      minSlot = s;
    }
  }
  *minSlot = {lock, level, value};
}

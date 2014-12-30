#include "TransTable.hpp"
#include <stdio.h>
#include <string.h>

#define SLOT_BITS 30
#define RES_BITS 4
#define N 8

constexpr uint64_t SIZE = (1ull << SLOT_BITS) - (1ull << (SLOT_BITS - RES_BITS));
constexpr uint64_t MASK = (1ull << SLOT_BITS) - 1;

TransTable::TransTable() :
  slots(new Slot[SIZE + N - 1])
{
  printf("Size %.2f GB, slot size %ld, info %ld\n",
         SIZE * sizeof(Slot) / (1024 * 1024 * 1024.0f), sizeof(Slot), sizeof(NodeInfo));
}

TransTable::~TransTable() {
  delete[] slots;
  slots = 0;
}

inline uint64_t getPos(uint128_t hash) {
  uint64_t pos = hash & MASK;
  while (pos >= SIZE) {
    pos >>= RES_BITS;
  }
  return pos;
}

inline uint64_t getLock(uint128_t hash) {
  return ((uint64_t) (hash >> 64)) & ((1ull << LOCK_BITS) - 1);
}

NodeInfo TransTable::lookup(uint128_t hash) {
  uint64_t pos = getPos(hash);
  uint64_t lock = getLock(hash);
  
  Slot buf[N];
  for (Slot *begin = slots + pos, *s = begin, *end = begin + N, *p = buf + 1; s < end; ++s, ++p) {
    if (s->lock == lock) {
      if (s == begin) {
        return s->info();
      } else {
        *buf = *s;
        memmove(begin, buf, (p - buf) * sizeof(Slot));
        return buf->info();
      }
    } else if (s->isEmpty()) {
        break;
    } else {
      *p = *s;
    }
  }
  return {-TOTAL_POINTS, TOTAL_POINTS};
} 

void TransTable::set(uint128_t hash, int min, int max) {
  uint64_t pos = getPos(hash);
  uint64_t lock = getLock(hash);

  Slot buf[N];
  Slot *p = buf + 1;
  Slot *begin = slots + pos;
  for (Slot *begin = slots + pos, *s = begin, *end = begin + N - 1; s < end; ++s, ++p) {
    if (s->isEmpty() || s->lock == lock) {
      break;
    } else {
      *p = *s;
    }
  }
  buf[0] = {lock, (signed char) min, (signed char) max};
  memmove(begin, buf, (p - buf) * sizeof(Slot));
}

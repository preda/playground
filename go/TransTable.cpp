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
         SIZE * sizeof(Slot) / (1024 * 1024 * 1024.0f), sizeof(Slot), sizeof(SlotInfo));
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

inline SlotInfo makeInfo(Slot *s) {
  return {s->pos, s->min, s->max};
}

SlotInfo TransTable::lookup(uint128_t hash) {
  return lookup(getPos(hash), getLock(hash));
}

SlotInfo TransTable::lookup(uint64_t pos, uint64_t lock) {
  Slot buf[N];
  for (Slot *begin = slots + pos, *s = begin, *end = begin + N, *p = buf + 1; s < end; ++s, ++p) {
    if (s->lock == lock) {
      if (s == begin) {
        return makeInfo(s);
      } else {
        *buf = *s;
        memmove(begin, buf, (p - buf) * sizeof(Slot));
        return makeInfo(buf);
      }
    } else if (s->isEmpty()) {
        break;
    } else {
      *p = *s;
    }
  }
  return SlotInfo();
} 

void TransTable::set(uint64_t pos, uint64_t lock, SlotInfo info) {
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
  buf[0] = {lock, info.pos, info.min, info.max};
  memmove(begin, buf, (p - buf) * sizeof(Slot));
}

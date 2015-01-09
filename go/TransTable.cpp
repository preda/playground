// (c) Copyright 2014 Mihai Preda. All rights reserved.

#include "TransTable.hpp"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define LOCK_BITS 48
#define SLOT_BITS 30
#define RES_BITS 2
#define SEARCH 8

constexpr uint64_t SIZE = (1ull << SLOT_BITS) - (1ull << (SLOT_BITS - RES_BITS));
constexpr uint64_t MASK = (1ull << SLOT_BITS) - 1;
constexpr uint64_t LOCK_MASK = (1ul << LOCK_BITS) - 1;

TransTable::TransTable() :
  slots((uint64_t *) calloc(SIZE + SEARCH - 1, 8))
{
  printf("Size %.2f GB\n", SIZE * 8 / (1024 * 1024 * 1024.0f));
}

TransTable::~TransTable() {
  free(slots);
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
  return ((uint64_t) (hash >> 64)) & LOCK_MASK;
}

std::tuple<int, bool> TransTable::get(uint128_t hash, int depth) {
  uint64_t pos = getPos(hash);
  uint64_t lock = getLock(hash);
  
  uint64_t buf[SEARCH];
  for (uint64_t *begin = slots + pos, *s = begin, *end = begin + SEARCH, *p = buf;
       s < end; ++s) {
    uint64_t v = *s;
    if (v == 0) {
      break;
    } else if ((v & LOCK_BITS) == lock) {
      if (s > begin) {
        *begin = v;
        memmove(begin + 1, buf, (p - buf) * 8);
      }
      int bound = (signed char) (v >> LOCK_BITS);
      bool exact = bound & 1;
      bound >>= 1;
      if (bound != UNKNOWN) {
        return std::make_tuple(bound, exact);
      }
      int d = (signed char) (v >> (LOCK_BITS + 8));      
      if (depth <= d) {
        return std::make_tuple(UNKNOWN, false);
      } else {
        break;
      }
    } else {
      *p++ = v;
    }
  }
  return std::make_tuple(TT_NOT_FOUND, false);
} 

void TransTable::set(uint128_t hash, int depth, int bound, bool exact) {
  uint64_t pos = getPos(hash);
  uint64_t lock = getLock(hash);

  uint64_t buf[SEARCH];
  uint64_t *p = buf;
  uint64_t *begin = slots + pos;
  for (uint64_t *s = begin, *end = begin + SEARCH - 1;
       s < end; ++s) {
    uint64_t v = *s;
    if (v == 0 || ((v & LOCK_BITS) == lock)) {
      break;
    } else {
      *p++ = v;
    }
  }
  bound = (bound << 1) | (exact ? 1 : 0);
  *begin = (((uint64_t) (unsigned char) depth) << (LOCK_BITS + 8))
        | (((uint64_t) (unsigned char) bound) << LOCK_BITS)
        | lock;
  if (p > buf) {
    memmove(begin + 1, buf, (p - buf) * 8);
  }
}

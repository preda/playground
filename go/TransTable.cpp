// (c) Copyright 2014 Mihai Preda. All rights reserved.

#include "TransTable.hpp"
#include "transtable.hpp"
#include "Hash.hpp"
#include "Value.hpp"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

TransTable::TransTable() :
  slots((uint64_t *) calloc(SIZE + SEARCH - 1, 8))
{
  printf("Size %.2f GB\n", SIZE * 8 / (1024 * 1024 * 1024.0f));
}

TransTable::~TransTable() {
  free(slots);
  slots = 0;
}

Value TransTable::get(const Hash &hash, int d) {
  uint64_t pos  = hash.pos;
  uint64_t lock = hash.lock;
  uint64_t v = slots[pos];
  if ((v & LOCK_MASK) != lock) { return Value::makeNoInfo(); }
  unsigned packed = v >> LOCK_BITS;
  int depth = packed & 0x3f;
  if (depth < d) { return Value::makeNoInfo(); }  
  int kind = ((packed >> 6) & 0x3) + 1;
  int value = (sbyte) ((packed >> 8) & 0xff);
  return Value(kind, value);
}

void TransTable::set(const Hash &hash, Value v, int depth) {
  if (depth >= v.getHistoryPos()) {
    assert(v.kind > 0 && v.kind <= 4);
    assert(depth >= 0 && depth < 64);
    unsigned packed = (((unsigned)(byte) v.value) << 8) | ((v.kind - 1) << 6) | depth;    
    uint64_t pos  = hash.pos;
    uint64_t lock = hash.lock;
    slots[pos] = (((uint64_t) packed) << LOCK_BITS) | lock;
  }
}

/* 
  uint64_t buf[SEARCH];
  for (uint64_t *begin = slots + pos, *s = begin, *end = begin + SEARCH, *p = buf;
       s < end; ++s) {
    uint64_t v = *s;
    if (v == 0) {
      break;
    } else if ((v & LOCK_MASK) == lock) {
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
*/

/*
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
*/

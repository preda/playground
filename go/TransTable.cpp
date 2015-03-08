// (c) Copyright 2014 Mihai Preda. All rights reserved.

#include "TransTable.hpp"
#include "Hash.hpp"
#include "Value.hpp"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

TransTable::TransTable() :
  pages((Page *) malloc(NPAGE * sizeof(Page)))
{
  printf("Size %.2f GB\n", (NPAGE * sizeof(Page)) / (1024 * 1024 * 1024.0f));
}

TransTable::~TransTable() {
  free(slots);
  slots = 0;
}

Value TransTable::get(const Hash &hash, int depth, int beta) {
  uint64_t h = hash.hash;
  int pageId = h & PAGE_MASK;
  if (!active.get(pageId)) {
    return Value::makeNone();
  }
  uint64_t *page = pages[pageId].slots;
  unsigned lock = (unsigned)(hash >> NPAGE_BITS);
  for (uint64_t *p = page, *end = page + SLOTS_PER_PAGE; p < end; ++p) {
    if ((unsigned)(*p) == lock) {
      unsigned packed = *p >> 32;
      int value = (sbyte) (packed & 0xff);
      int     d = (packed >> 8) & 0x3f;
      bool isLow = packed & (1 << 14);
      bool isUpp = packed & (1 << 15);
      
      if (isLow || isUpp) {
        return Value(isLow, isUpp, value);
      } else {
        return (value == beta && d >= depth) ? Value::makeDepthLimited() : Value::makeNone();
      }
    }
  }
  return Value::makeNone();

  
  uint64_t *page = slots + (h & PAGE_MASK) * N_PER_PAGE;
  
  uint64_t pos  = hash.pos;
  uint64_t lock = hash.lock;
  uint64_t v = slots[pos];
  if ((v & LOCK_MASK) != lock) { return Value::makeNone(); }
  
  unsigned packed = v >> LOCK_BITS;
  int low = (sbyte)(packed & 0xff);
  int upp = (sbyte)((packed >> 8) & 0xff);
  return Value(low, upp);
}

void TransTable::set(const Hash &hash, Value value, int depth, int beta) {
  assert(depth >= 0 && depth <= 0x3f);
  assert(-128 <= beta && beta < 128);
  if (depth >= value.historyPos && !value.isDepthLimited()) {
    // assert(value.isEnough(beta));
    unsigned packed;
    packed = ((byte)value.low) | (((byte)value.upp) << 8);
    slots[hash.pos] = (((uint64_t) packed) << LOCK_BITS) | hash.lock;
  }
}

    /*
    if (value.isDepthLimited()) {
      packed = (depth << 8) | (byte)beta;
    } else {
    */

  // return (value == beta && d >= depth) ? Value::makeDepthLimited() : Value::makeNone();

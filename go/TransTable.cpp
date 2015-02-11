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

Value TransTable::get(const Hash &hash, int depth, int beta) {
  uint64_t pos  = hash.pos;
  uint64_t lock = hash.lock;
  uint64_t v = slots[pos];
  if ((v & LOCK_MASK) != lock) { return Value::makeNone(); }
  
  unsigned packed = v >> LOCK_BITS;
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

void TransTable::set(const Hash &hash, Value value, int depth, int beta) {
  assert(depth >= 0 && depth <= 0x3f);
  assert(-128 <= beta && beta < 128);
  if (depth >= value.historyPos) {
    unsigned packed;
    assert(value.isEnough(beta) || value.isDepthLimited());
    if (value.isDepthLimited()) {
      assert(!value.isLow && !value.isUpp);
      packed = (depth << 8) | (byte)beta;
    } else {
      assert(value.isLow || value.isUpp);
      packed = (value.isLow ? (1<<14) : 0) | (value.isUpp ? (1<<15) : 0) | (byte)value.value;      
    }
    slots[hash.pos] = (((uint64_t) packed) << LOCK_BITS) | hash.lock;
  }
}

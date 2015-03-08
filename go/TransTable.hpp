// (c) Copyright 2014 Mihai Preda. All rights reserved.

#pragma once

#include "go.hpp"

#define NPAGE_BITS 21
#define NPAGE (1 << NPAGE_BITS)
#define PAGE_MASK (NPAGE - 1)
#define SLOTS_PER_PAGE 512

// #define LOCK_BITS 32



/*
#define LOCK_BITS 48
#define SLOT_BITS 29
#define RES_BITS SLOT_BITS
#define SEARCH 8

constexpr uint64_t SIZE = (1ull << SLOT_BITS) - (1ull << (SLOT_BITS - RES_BITS));
constexpr uint64_t MASK = (1ull << SLOT_BITS) - 1;
constexpr uint64_t LOCK_MASK = (1ull << LOCK_BITS) - 1;
*/

class Hash;
class Value;

struct Page {
  uint64_t slots[SLOTS_PER_PAGE];
};

class TransTable {
private:
  std::bitset<NPAGE> active;
  Page *pages;

public:
  TransTable();
  ~TransTable();

  Value get(const Hash &hash, int depth, int beta);
  void set(const Hash &hash, Value value, int depth, int beta);
};

// (c) Copyright 2014 Mihai Preda. All rights reserved.

#pragma once

#define LOCK_BITS 48
#define SLOT_BITS 30
#define RES_BITS 2
#define SEARCH 8

constexpr uint64_t SIZE = (1ull << SLOT_BITS) - (1ull << (SLOT_BITS - RES_BITS));
constexpr uint64_t MASK = (1ull << SLOT_BITS) - 1;
constexpr uint64_t LOCK_MASK = (1ul << LOCK_BITS) - 1;

// Copyright (c) Mihai Preda 2013-2014

#pragma once

#include "go.hpp"
#include <tuple>

class TransTable {
private:
  uint64_t *slots;

public:
  TransTable();
  ~TransTable();

  std::tuple<int, bool> get(uint128_t hash, int depth);
  void set(uint128_t hash, int depth, int bound, bool exact);
};

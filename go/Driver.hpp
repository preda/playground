// Copyright (c) Mihai Preda 2013-2014

#pragma once

#include "Node.hpp"
#include "TransTable.hpp"
#include <unordered_set>

class Driver {
  struct HashHasher { size_t operator()(uint128_t key) const { return (size_t) key; } };
  std::unordered_set<uint128_t, HashHasher> history;
  TransTable tt;
  
public:
  template<int C> int AB(const Node &n, int beta, int d);

  int mtdf(int f, int d);
};

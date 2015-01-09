// (c) Copyright 2014 Mihai Preda. All rights reserved.

#pragma once

#include "Node.hpp"
#include "TransTable.hpp"

class Driver {
  TransTable tt;
  
public:
  int MAX(const Node &n, const int beta, int d);
  int MIN(const Node &n, const int beta, int d);

  void mtd();
};

/*
  struct HashHasher { size_t operator()(uint128_t key) const { return (size_t) key; } };
  std::unordered_set<uint128_t, HashHasher> history;
  template<int C> int AB(const Node &n, int beta, int d);
*/

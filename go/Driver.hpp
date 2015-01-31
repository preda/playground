// (c) Copyright 2014 Mihai Preda. All rights reserved.

#pragma once

#include "Node.hpp"
#include "TransTable.hpp"

#include <vector>

class History;

class Driver {
  TransTable tt;
  Node stack[64];
  int minD;
  std::vector<int> minMoves;
  
public:
  template<bool MAX> Value miniMax(const Node &, const Hash &, History *, const int beta, int d);
  template<bool MAX> int extract(const Node &, const Hash &, History *, const int beta, int d, int limit, std::vector<int> &moves);
  void mtd();
};

// (c) Copyright 2014 Mihai Preda. All rights reserved.

#pragma once

#include "Node.hpp"
#include "TransTable.hpp"

#include <vector>

class History;

class Driver {
  TransTable tt;
  // Node interestNode;
  
public:
  Value miniMax(const Node &, const Hash &, History *, const int beta, int d);
  void mtd(const Node &n, int depth);
};

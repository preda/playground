// (c) Copyright 2014 Mihai Preda. All rights reserved.

#pragma once

#include "Node.hpp"
#include "TransTable.hpp"

class History;

class Driver {
  TransTable tt;
  
public:
  template<bool MAX> Value miniMax(const Node &, const Hash &, History *, const int beta, int d);
  void mtd();
};

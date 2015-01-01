#pragma once

#include "Node.hpp"
#include "TransTable.hpp"
#include <unordered_map>
#include <algorithm>

class Driver {
 public:
  // std::unordered_map<uint128_t> history;
  TransTable tt;
  template<int C> int AB(Node *n, int beta, int d);  
};

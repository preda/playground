#pragma once

#include "Node.hpp"
#include "TransTable.hpp"
#include <unordered_set>
#include <algorithm>

struct HistoryHasher {
  size_t operator()(uint128_t key) const { return (size_t) key; }
};

HistoryHasher historyHasher;

class Driver {
  std::unordered_set<uint128_t, HistoryHasher> history;
  TransTable tt;
  
public:

  template<int C> int AB(const Node &n, int beta, int d);  
};

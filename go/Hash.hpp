// (c) Copyright 2014 Mihai Preda. All rights reserved.

#pragma once

#include "go.hpp"

// template<bool BLACK> uint128_t hashUpdate(int pos, int oldKoPos, int koPos, int oldNPass, int nPass, uint64_t capture);

class Hash {
  Hash(uint128_t h);
  
public:
  uint128_t hash;
  uint64_t pos, lock;

  Hash() : Hash(-1) { }

  template<bool BLACK> Hash update(int pos, int oldNPass, int nPass, uint64_t capture) const;
  void print();
};

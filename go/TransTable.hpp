// (c) Copyright 2014 Mihai Preda. All rights reserved.

#pragma once

#include "go.hpp"

class Hash;
class Value;

class TransTable {
private:
  uint64_t *slots;

public:
  TransTable();
  ~TransTable();

  Value get(const Hash &hash);
  void set(const Hash &hash, Value value, int depth);
};

// (c) Copyright 2014 Mihai Preda. All rights reserved.

#pragma once

#include "go.hpp"

#include <assert.h>
#include <algorithm>
#include <stdio.h>

class Value {
public:
  sbyte low, upp;
  byte historyPos;
  
  Value(int low, int upp, int historyPos = 0) :
    low((sbyte)low),
    upp((sbyte)upp),
    historyPos((byte) historyPos)
  {
    assert(low >= -N-1 && low <= N && upp >= -N && upp <= N && low <= upp);
  }

  void print() {
    if (historyPos) { printf("history %d ", historyPos); }
    if (isNone()) {
      printf("none\n");
    } else if (isDepthLimited()) {
      printf("depth\n");
    } else {
      printf("[%d, %d]\n", low, upp);
    }
  }

  bool isDepthLimited() { return low == -N-1; }
  bool isNone() { return low == -N && upp == N; }
  bool isEnough(int beta) { return isCut<true>(beta) || isCut<false>(beta); }
  template<bool MAX> bool isCut(int beta) { return MAX ? low >= beta : (upp < beta); }
  
  template<bool MAX> Value accumulate(Value o) {
    assert(!isNone() && !o.isNone());
    int histPos = std::max(historyPos, o.historyPos);
    if (isDepthLimited() || o.isDepthLimited()) { return Value::makeDepthLimited(histPos); }
    return MAX ? Value(std::max(low, o.low), std::max(upp, o.upp), histPos) :
      Value(std::min(low, o.low), std::min(upp, o.upp), histPos);
  }

  static Value makeUpp(int v)     { return Value(-N, v); }
  static Value makeLow(int v)     { return Value(v,  N); }
  static Value makeExact(int v)   { return Value(v,  v); }
  static Value makeDepthLimited(int histPos = 0) { return Value(-N-1, N, histPos); }
  static Value makeNone()         { return Value(-N, N); }

  template<bool MAX> Value relaxBound() const {
    return MAX ? Value(low, N) : Value(-N, upp);
  }

  void updateHistoryPos(int hp) {
    assert(hp > 0);
    historyPos = (byte) std::max<int>(historyPos, hp);
  }
};

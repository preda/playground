// (c) Copyright 2014 Mihai Preda. All rights reserved.

#pragma once

#include "go.hpp"

#include <assert.h>
#include <algorithm>
#include <stdio.h>

enum {
  LOW = 1,
  UPP = 2,
};

class Value {
public:
  bool isLow;
  bool isUpp;
  sbyte value;
  byte historyPos;
  
  Value(bool low, bool upp, int value, int historyPos = 0) :
    isLow(low),
    isUpp(upp),
    value((sbyte) value),
    historyPos((byte) historyPos)
  {
    if (!isLow && !isUpp) {
      assert(value == 0 || value == -N-1);
    } else {
      assert(isLow || value <  N);
      assert(isUpp || value > -N);
    }
  }

  void print() {
    if (historyPos) { printf("history %d ", historyPos); }
    if (isUpp || isLow) {
      const char *sign = (isUpp && isLow) ? "==" : (isUpp ? "<=" : ">=");
      printf("value %s%d\n", sign, (int) value);
    } else {
      if (value == 0) {
        printf("value depth limited\n");
      } else {
        assert(value == -N-1);
        printf("value none\n");
      }
    }
  }

  bool isDepthLimited() { return !isUpp && !isLow && value == 0; }
  bool isNone() { return !isUpp && !isLow && value == -N-1; }
  bool isEnough(int beta) { return isCut<true>(beta) || isCut<false>(beta); }
  template<bool MAX> bool isCut(int beta) { return MAX ? isLow && value >= beta : (isUpp && value < beta); }
  
  template<bool MAX> Value accumulate(Value o) {
    assert(!isNone() && !o.isNone());
    int histPos = std::max(historyPos, o.historyPos);
    if (isDepthLimited() || o.isDepthLimited()) { return Value::makeDepthLimited(histPos); }
    if (MAX) {
      assert(isUpp && o.isUpp);
      int v = std::max(value, o.value);
      bool newIsLow = (isLow && value == v) || (o.isLow && o.value == v);
      return Value(newIsLow, true, v, histPos);      
    } else {
      assert(isLow && o.isLow);
      int v = std::min(value, o.value);
      bool newIsUpp = (isUpp && value == v) || (o.isUpp && o.value == v);
      return Value(true, newIsUpp, v, histPos);
    }
  }

  static Value makeUpp(int v)     { return Value(false, true,  v); }
  static Value makeLow(int v)     { return Value(true,  false, v); }
  static Value makeExact(int v)   { return Value(true,  true,  v); }
  static Value makeDepthLimited(int histPos = 0) { return Value(false, false, 0, histPos); }
  static Value makeNone()         { return Value(false, false, -N-1); }

  template<bool MAX> Value relaxBound() const {
    assert(isLow || isUpp);
    return (isLow && isUpp) ? Value(MAX, !MAX, value, historyPos) : *this;
  }

  void updateHistoryPos(int hp) {
    assert(hp > 0);
    historyPos = (byte) std::max<int>(historyPos, hp);
  }
};

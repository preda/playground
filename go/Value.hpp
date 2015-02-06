// (c) Copyright 2014 Mihai Preda. All rights reserved.

#pragma once

#include "go.hpp"
#include <assert.h>
#include <algorithm>
#include <stdio.h>

  enum {
    UPPER_BOUND = 1,
    LOWER_BOUND = 2,
    EXACT = 3,
    UNKNOWN = 4,
  };


class Value {
public:
  byte kind;
  sbyte value;
  byte historyPos;
  
  Value(int kind, int value, int historyPos = 0) :
    kind((byte) kind),
    value((sbyte) value),
    historyPos((byte) historyPos)
  { }

  void print() {
    const char *labels[] = {"RESERVED", "UPPER_BOUND", "LOWER_BOUND", "EXACT", "UNKNOWN"};
    printf("%s %d %d\n", labels[kind], (int)value, (int)historyPos);
  }
  
  template<bool MAX> Value accumulate(Value o) {
    int histPos = std::max(historyPos, o.historyPos);
    if (MAX) {
      assert(kind != LOWER_BOUND && o.kind != LOWER_BOUND);
    } else {
      assert(kind != UPPER_BOUND && o.kind != UPPER_BOUND);
    }
    int v = MAX ? std::max(value, o.value) : std::min(value, o.value);
    int k = (kind == UNKNOWN || o.kind == UNKNOWN) ? UNKNOWN :
      ((kind == EXACT && value == v) || (o.kind == EXACT && o.value == v)) ? EXACT :
      (MAX ? UPPER_BOUND : LOWER_BOUND);
    return Value(k, v, histPos);
  }

  void updateHistoryPos(int hp) {
    historyPos = (byte) std::max<int>(historyPos, hp);
  }
  
  static Value makeExact(int v) { return Value(EXACT, v); }
  static Value makeUpperBound(int v) { return Value(UPPER_BOUND, v); }
  static Value makeLowerBound(int v) { return Value(LOWER_BOUND, v); }
  static Value makeUnknown(int bound) { return Value(UNKNOWN, bound, 0); }
  static Value makeNoInfo() { return makeUnknown(N); }

  int getKind() { return kind; }
  int getValue() { return value; }
  int getHistoryPos() { return historyPos; }
  
  bool unknownAt(int beta) {
    return kind == UNKNOWN && value < beta;
  }

  bool noInfoAt(int beta) {
    return kind == UNKNOWN && value >= beta;
  }

  bool isEnough(int beta) {
    return isCut<true>(beta) || isCut<false>(beta) || unknownAt(beta);
  }

  template<bool MAX> bool isCut(int beta) {
    return MAX ? value >= beta && (kind == LOWER_BOUND || kind == EXACT) :
      (value < beta && (kind == UPPER_BOUND || kind == EXACT));
  }

  template<bool MAX> Value relaxBound() const {
    assert(kind != UNKNOWN);
    if (kind != EXACT) { return *this; }
    assert(kind == EXACT);
    return Value(MAX ? LOWER_BOUND : UPPER_BOUND, value, historyPos);
  }  
};

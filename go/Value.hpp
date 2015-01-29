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
private:

  sbyte value;
  byte kind;
  byte depth;
  byte historyPos;
  
  Value(int kind, int value, int depth, int historyPos) :
    value((sbyte) value),
    kind((byte) kind),
    depth((byte) depth),
    historyPos((byte) historyPos) { }

  Value(int kind, int value) : Value(kind, value, 0, 0) {}

public:
  void print() {
    const char *labels[] = {"UPPER_BOUND", "LOWER_BOUND", "EXACT", "UNKNOWN"};
    printf("%s %d %d %d\n", labels[kind], (int)value, (int)depth, (int)historyPos);
  }
  
  template<bool MAX> Value accumulate(Value o) {
    int histPos = std::max(historyPos, o.historyPos);
    // if (histPos <= d) { histPos = 0; }
    if (MAX) {
      assert(kind != LOWER_BOUND && o.kind != LOWER_BOUND);
    } else {
      assert(kind != UPPER_BOUND && o.kind != UPPER_BOUND);
    }
    int v = MAX ? std::max(value, o.value) : std::min(value, o.value);
    int k = (kind == UNKNOWN || o.kind == UNKNOWN) ? UNKNOWN :
      ((kind == EXACT && value == v) || (o.kind == EXACT && o.value == v)) ? EXACT :
      (MAX ? UPPER_BOUND : LOWER_BOUND);
    return Value(k, v, 0, histPos);
  }

  void updateHistoryPos(int hp) {
    historyPos = (byte) std::max<int>(historyPos, hp);
  }
  
  static Value makeExact(int v) { return Value(EXACT, v); }
  static Value makeUpperBound(int v) { return Value(UPPER_BOUND, v); }
  static Value makeLowerBound(int v) { return Value(LOWER_BOUND, v); }
  static Value makeUnknown(int bound) { return Value(UNKNOWN, bound, 1, 0); }

  unsigned pack(int depth) {
    assert(kind > 0 && kind <= 4);
    assert(depth < 64);
    return (((unsigned)(byte) value) << 8) | ((kind - 1) << 6) | depth;
  }
  
  static Value unpack(unsigned packed) {
    return Value(((packed >> 6) & 0x3) + 1, (sbyte) ((packed >> 8) & 0xff), packed & 0x3f, 0);
  }

  int getKind() { return kind; }
  int getValue() { return value; }
  int getDepth() { return depth; }
  int getHistoryPos() { return historyPos; }
  
  bool unknownAt(int beta, int d) {
    return (kind == UNKNOWN) && value < beta && depth > d;
  }

  bool noInfoAt(int beta, int d) {
    return (kind == UNKNOWN) && (value >= beta || depth <= d);
  }
  
  /*
  int lowBound() {
    assert(kind == UPPER_BOUND || kind == LOWER_BOUND || kind == EXACT);
    return (kind == LOW || kind == EXACT) ? value : -N;
  }

  int upBound() {
    assert(kind == UP_BOUND || kind == LO_BOUND || kind == EXACT);
    return (kind == UP || kind == EXACT) ? value : N;
  }
  */

  bool isEnough(int beta, int d) {
    return isCut<true>(beta) || isCut<false>(beta) || unknownAt(beta, d);
  }

  template<bool MAX> bool isCut(int beta) {
    return MAX ? value >= beta && (kind == LOWER_BOUND || kind == EXACT) :
      value < beta && (kind == UPPER_BOUND || kind == EXACT);
  }

  template<bool MAX> Value relaxBound() const {
    assert(kind != UNKNOWN);
    int k = (kind == EXACT) ? (MAX ? LOWER_BOUND : UPPER_BOUND) : kind;
    return Value(k, value, depth, historyPos);
  }
  
};

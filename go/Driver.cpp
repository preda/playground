// (c) Copyright 2014 Mihai Preda. All rights reserved.

#include "Driver.hpp"
#include "Hash.hpp"
#include "Value.hpp"

#include <stdio.h>
#include <assert.h>
#include <unordered_map>

struct HistHash {
  inline size_t operator()(uint128_t key) const { return (size_t) key; }
};

class History {
  std::unordered_map<uint128_t, int, HistHash> map;

public:
  int depthOf(const Hash &hash) {
    auto it = map.find(hash.hash);
    return it == map.end() ? 0 : it->second;
  }
  
  void push(const Hash &hash, int d) {
    bool added = map.emplace(hash.hash, d).second;
    assert(added);
  }

  void pop(const Hash &hash) {
    map.erase(hash.hash);
    // assert(n == 1);
  }
};

template<typename T> class Post {
private:
  T f;
public:
  Post(T f) : f(f) {}
  ~Post() { f(); }
};

void Driver::mtd() {
  Node root;
  Hash hash;
  History history;
  int beta = N;
  int d = 10;
  while (true) {
    Value v = miniMax<true>(root, hash, &history, beta, d);
    printf("MTDF %d, beta %d: ", d, beta);
    v.print();
    int value = v.getValue();
    if (v.unknownAt(beta, d)) {
      assert(v.getDepth() == d + 1);
      assert(value < beta);
      if (value > -N) {
        beta = value;
      } else {
        d += 2;
      }
    } else {
      int kind = v.getKind();
      if (kind == EXACT) {
        break;
      } else if (kind == LOWER_BOUND) {
        assert(value == beta);
        break;
      } else {
        assert(kind == UPPER_BOUND);
        assert(value < beta);
        beta = value;
      }
    }
  }
}

template<bool MAX>
Value Driver::miniMax(const Node &n, const Hash &hash, History *history, const int beta, int d) {
  Value v = tt.get(hash);
  if (v.isEnough(beta, d)) {
    return v;
  }
  if (v.noInfoAt(beta, d)) {
    v = n.score(beta);
    if (v.isEnough(beta, d)) {
      return v;
    }
  }
  assert(v.getKind() == LOWER_BOUND);
  int value = MAX ? v.getValue() : N;
  // printf("d %d v %d\n", d, value);
  Value acc = Value::makeExact(value);
  
  Vect<byte, N> moves;
  n.genMoves<MAX ? BLACK : WHITE>(moves);
  int nMoves = moves.size();
  Hash hashes[nMoves];
  uint64_t done = 0;
  int historyDepth = 0;
  
  for (int i = 0; i < nMoves; ++i) {
    int p = moves[i];
    Hash h = n.hashOnPlay<MAX ? BLACK : WHITE>(hash, p);
    hashes[i] = h;
    int hd = history->depthOf(h);
    if (hd) {
      assert(hd > d);
      historyDepth = std::max(historyDepth, hd);
      SET(p, done);
    } else {    
      Value v = tt.get(h);
      if (v.isCut<MAX>(beta)) {
        return v.relaxBound<MAX>();
      }
      if (v.isEnough(beta, d - 1)) {
        acc = acc.accumulate<MAX>(v);
        SET(p, done);
      }
    }
  }

  if (d == 0) {
    acc = Value::makeUnknown(acc.getValue()); //, historyDepth);
  } else {
    history->push(hash, d);  
    // Post onReturn([const& hash]() { history->pop(hash); }
    for (int i = 0; i < nMoves; ++i) {
      int p = moves[i];
      if (!IS(p, done)) {
        Hash h = hashes[i];
        Node sub = n.play<MAX ? BLACK : WHITE>(p);
        Value v = miniMax<!MAX>(sub, h, history, beta, d - 1);
        history->pop(h);
        tt.set(h, v, d - 1);
        // sub.print(); v.print();
        if (v.isCut<MAX>(beta)) {
          return v.relaxBound<MAX>();
        }
        acc = acc.accumulate<MAX>(v);
      }
    }
  }
  acc.updateHistoryPos(historyDepth);
  return acc;
}

int main(int argc, char **argv) {
  Driver driver;
  driver.mtd();
}

    /*
    if (v.kind == UNKNOWN) {
      hasUnknown = true;
      unknownBound = std::max(unknownBound, v.value);
    } else if (v.isMaxCut(beta)) {
      Value vv = Value.makeLowerBound(v);
      tt.set(hash, vv);
      return vv;
    } else {
      assert(v.kind != LOW);
      max = std::max(max, v.value);
    }

    
  if (hasUnknown) {
    Value v = Value.makeUnknown(std::max(unknownBound, max), d);
    tt.set(hash, v);
    return v;
  } else {
    Value v = Value.makeUpperBound(max);
    tt.set(hash, d, max, false);
    return max;
  }
    */
/*
int Driver::MIN(const Node &n, int beta, int d) {
  uint128_t hash = n.getHash();
  int bound;
  bool exact;
  std::tie(bound, exact) = tt.get(hash, d);
  if (bound == TT_NOT_FOUND) {
    assert(!exact);
    int max;
    std::tie(bound, max) = n.score();
    exact = bound == max;
    tt.set(hash, d, bound, exact);
  }
  if (exact || bound == UNKNOWN || bound >= beta) {
    return bound;
  }
  Vect<byte, N> moves;
  uint64_t unknown = 0;
  n.genMoves<WHITE>(moves);
  for (int p : moves) {
    int s = std::get<0>(tt.get(n.hashOnPlay<WHITE>(p), d - 1));
    if (s == TT_NOT_FOUND) { continue; }
    if (s == UNKNOWN) {
      SET(p, unknown);
    } else if (s < beta) {
      tt.set(hash, d, s, false);
      return s;
    }
  }
  int min = N;
  for (int p : moves) {
    if (!IS(p, unknown)) {
      int s = MAX(n.play<WHITE>(p), beta, d - 1);
      if (s == UNKNOWN) {
        min = UNKNOWN;
      } else if (s < beta) {
        tt.set(hash, d, s, false);
        return s;
      } else if (min != UNKNOWN && s < min) {
        min = s;
      }
    }
  }
  if (unknown) { min = UNKNOWN; }
  tt.set(hash, d, min, min != UNKNOWN && min >= beta);
  return min;
}
*/

/*
inline int negaUnknown(int g) { return (g == UNKNOWN) ? UNKNOWN : -g; }

inline int maxUnknown(int a, int b) {
  return (a == UNKNOWN || b == UNKNOWN) ? UNKNOWN : std::max(a, b);
}


  template<int C> int Driver::AB(const Node &n, int beta, int d) {
  assert(C == (d & 1));  
  // n.print();
  uint128_t hash = n.getHash();
  int bound;
  bool exact;
  std::tie<bound, exact> = tt.get<C>(hash, d);
  if (exact || bound == UNKNOWN ||
      (C == BLACK && bound < beta) ||
      (C == WHITE && bound >= beta)) {
    return bound;
  }

  if ((C == BLACK && bound == N) || (C == WHITE && bound == -N)) {
    ScoreBounds score = n.score<C>();
    bool exact = score.min == score.max;
    int sc = C == BLACK ? score.max : score.min;
    if (exact ||
        (C == BLACK && sc <  beta) ||
        (C == WHITE && sc >= beta)) {
      tt.set(hash, d, sc, exact);
      return sc;
    }
  }    
    
  Vect<byte, N> moves;
  n.genMoves<C>(moves);
  // printf("%d beta %d Moves %d\n", d, beta, moves.size());
  int subBeta = -(beta - 1);
  for (int p : moves) {
    int s = -tt.lookup(n.hashOnPlay<C>(p)).max;
    if (s >= beta) {
      // printf("%d beta %d ETC: %d\n", d, beta, s);
      tt.set(hash, s, max);      
      return s;
    }
  }

  if (d <= 0) { return UNKNOWN; }
  
  int g = -N;
  for (int p : moves) {
    int s = negaUnknown(AB<1-C>(n.play<C>(p), subBeta, d - 1));
    if (s >= beta) {
      // printf("%d beta %d Beta cut: %d\n", d, beta, s); 
      tt.set(hash, s, max);
      return s;
    }
    g = maxUnknown(g, s);
  }
  if (g != UNKNOWN) {
    tt.set(hash, min, g);    
  }
  // printf("A %d beta %d: %d\n", d, beta, g);
  return g;
}

template int Driver::AB<BLACK>(const Node&, int, int);
template int Driver::AB<WHITE>(const Node&, int, int);
*/


  /*
  bool added = history.insert(hash).second;
  if(!added && !n.lastWasPass()) { 
    return UNKNOWN; // -N
  }
  */


  /*
  auto f = [this, hash, added](){ if (added) { history.erase(hash); }};
  Post<decltype(f)> onReturn(f);
  */

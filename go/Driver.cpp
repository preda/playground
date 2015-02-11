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
    (void) added;
  }

  void pop(const Hash &hash, int d) {
    auto n = map.erase(hash.hash);    
    assert(n == 1);
    (void) n;
  }
};

static const int LIMIT = 35;

void Driver::mtd(const Node &root) {
  Hash hash;
  History history;
  int beta = N;
  int d = 14;
  
  while (true) {
    if (d >= LIMIT) {
      limitPrint = true;
    }
    Value v = miniMax<true>(root, hash, &history, beta, d);
    assert(v.isEnough(beta) || v.isDepthLimited());
    printf("MTD %d, beta %d: ", d, beta); v.print();
    if (v.isDepthLimited()) {
      ++d;
    } else {
      if (v.isUpp) {
        assert(v.value < beta);
        beta = v.value;
        if (beta == -N) { break; }
      } else {
        assert(v.isLow && v.value == beta);
        break;
      }
    }
  }

  /*
  minMoves.clear();
  std::vector<int> work;
  int l = extract<true>(root, hash, &history, beta, d, d + 1, work);
  assert(l <= d);
  assert((int)minMoves.size() == l);
  Node n = root;
  for (int i = 0; i < l; ++i) {
    int p = minMoves[i];
    n = ((i&1) == 0) ? n.play<BLACK>(p) : n.play<WHITE>(p);
    n.print();
  }
  */
}

template<bool MAX>
Value Driver::miniMax(const Node &n, const Hash &hash, History *history, const int beta, int d) {
  bool interestPrint = false; // = d >= 3;
  if (false && n == interest) { interestPrint = true; printf("Found interest at depth %d\n", d); }

  if (interestPrint) {
    printf("Enter %d max %d\n", d, (int)MAX);
    n.print();
  }
  
  assert(d >= 0);
  Value v = tt.get(hash, d, beta);
  if (v.isEnough(beta) || v.isDepthLimited()) { return v; }
  if (v.isNone()) {
    v = n.score<MAX>(beta);
    if (v.isEnough(beta)) {
      if (interestPrint) { n.print("score"); v.print(); }
      tt.set(hash, v, d, beta);
      return v;
    }
  }
  assert(!n.isEnded());
  assert(!v.isDepthLimited()); // && !v.isNone());
  // assert((MAX && v.isLow) || (!MAX && v.isUpp));
  Value acc = Value::makeExact(MAX ? -N : N);
  /*
  int value = MAX ? (v.kind == LOWER_BOUND ? v.value : -N) :
    (v.kind == UPPER_BOUND ? v.value : N);
  Value acc = Value::makeExact(value);
  */
  
  Vect<byte, N+1> moves;
  n.genMoves<MAX ? BLACK : WHITE>(moves);
  int nMoves = moves.size();
  assert(nMoves > 0);
  if (interestPrint) { printf("nMoves %d\n", nMoves); }
  Hash hashes[nMoves];
  uint64_t todo = 0;
  
  for (int i = 0; i < nMoves; ++i) {
    int p = moves[i];
    Hash h = n.hashOnPlay<MAX ? BLACK : WHITE>(hash, p);
    hashes[i] = h;
    int historyPos = history->depthOf(h);
    if (historyPos) {
      // printf("move %d hist %d\n", p, historyPos);
      assert(historyPos > d);
      acc.updateHistoryPos(historyPos);
      if (limitPrint || interestPrint) { printf("done history "); v.print(); }
    } else {
      Value v = tt.get(h, d - 1, beta);
      // printf("move %d: ", p); v.print();
      // if (print) { h.print(); v.print(); }
      if (v.isCut<MAX>(beta)) {
        Value vv = v.relaxBound<MAX>();
        if (limitPrint || interestPrint) {
          // printf("tt cut ");
          vv.print();
        }
        tt.set(hash, vv, d, beta);
        return vv;
      } else if (v.isDepthLimited() || v.isCut<!MAX>(beta)) {
        acc = acc.accumulate<MAX>(v);
        // printf("%d from tt %d max %d ", p, d, (int)MAX); v.print(); acc.print();
      } else {
        SET(p, todo);
      }
      /*
      if (!v.isEnough(beta)) {
        SET(p, todo);
      } else {
        if (v.isCut<MAX>(beta)) {
        } else {
          acc.accumulate<MAX>(v);
        }
      }
      */
    }
  }
  
  if (todo) {
    if (d == 0) {
      // assert(!MAX || acc.value < beta); // ??
      acc = Value::makeDepthLimited(acc.historyPos);
    } else {
      history->push(hash, d);  
      for (int i = 0; i < nMoves; ++i) {
        int p = moves[i];
        if (IS(p, todo)) {
          Hash h = hashes[i];
          Node sub = n.play<MAX ? BLACK : WHITE>(p);
          Value v = miniMax<!MAX>(sub, h, history, beta, d - 1);
          assert(!v.isNone());
          if (v.isCut<MAX>(beta)) {
            acc = v.relaxBound<MAX>();
            // printf("cut %d max %d: ", d, (int)MAX);
            // v.print();
            // acc.print();
            break;
          } else {
            acc = acc.accumulate<MAX>(v);
          }
        }
      }
      history->pop(hash, d);
    }
  }
  if (interestPrint /*|| (d == 0 && !acc.isDepthLimited())*/) {
    // n.print();
    acc.print(); // printf("nMoves %d %lx\n", nMoves, todo);
    // assert(false);
  }
  tt.set(hash, acc, d, beta);
  return acc;
}

int main(int argc, char **argv) {
  Node n;
  // n.setup("xx.xoxxx.oo.o.oo", 0, P(2, 0));
  Driver driver;
  driver.mtd(n);
}
/*
template<bool MAX>
int Driver::extract(const Node &n, const Hash &hash, History *history, const int beta, int d, int limit, std::vector<int> &moves) {
  assert(limit > 0);
  --limit;
  if (n.isEnded()) {
    assert(n.score(beta).value == beta);
    minMoves = moves;
    return 0;
  }
  if (limit == 0 || d == 0) {
    return 1;
  }
  Vect<byte, N+1> subMoves;
  n.genMoves<MAX ? BLACK : WHITE>(subMoves);
  int nMoves = subMoves.size();
  assert(nMoves > 0);
  history->push(hash, d);
  // int minP = 0;
  for (int i = 0; i < nMoves; ++i) {
    int p = subMoves[i];
    Hash h = n.hashOnPlay<MAX ? BLACK : WHITE>(hash, p);
    if (!history->depthOf(h)) {
      Node sub = n.play<MAX ? BLACK : WHITE>(p);
      Value v = miniMax<!MAX>(sub, h, history, beta, d - 1);
      assert(v.isEnough(beta));
      if (MAX) {
        assert(v.value <= beta);
      } else {
        assert(v.value >= beta);
      }
      if (v.value == beta) {
        moves.push_back(p);
        int subLimit = extract<!MAX>(sub, h, history, beta, d - 1, limit, moves);
        moves.pop_back();
        if (subLimit < limit) {
          limit = subLimit;
          if (limit == 0) { break; }
          // minP = p;
        }
      }
    }
  }
  history->pop(hash, d);
  // moves.push(minP);
  return limit + 1;  
}
*/

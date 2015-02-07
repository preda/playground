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
  interest.setup("x.xxoxxo.ooo.o.o", 0);
  
  while (true) {
    if (d >= LIMIT) {
      limitPrint = true;
    }
    Value v = miniMax<true>(root, hash, &history, beta, d);
    tt.set(hash, v, d);
    printf("MTD %d, beta %d ", d, beta);
    v.print();
    int value = v.getValue();
    if (v.unknownAt(beta)) {
      assert(value < beta);
      ++d;
      // break;
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
}

template<bool MAX>
Value Driver::miniMax(const Node &n, const Hash &hash, History *history, const int beta, int d) {
  bool interestPrint = false;
  if (n == interest) { interestPrint = true; printf("found interest %d\n", d); }
  Value v = tt.get(hash, d);
  if (v.noInfoAt(beta)) {
    v = n.score(beta);
  }
  if (v.isEnough(beta)) {
    if (limitPrint) {
      printf("isEnough ");
      v.print();
    }
    return v;
  }
  assert((v.kind == LOWER_BOUND && v.value < beta) ||
         (v.kind == UPPER_BOUND && v.value >= beta));
  int value = MAX ? (v.kind == LOWER_BOUND ? v.value : -N) :
    (v.kind == UPPER_BOUND ? v.value : N);
  Value acc = Value::makeExact(value);

  Vect<byte, N+1> moves;
  n.genMoves<MAX ? BLACK : WHITE>(moves);
  int nMoves = moves.size();
  assert(nMoves > 0);
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
      if (limitPrint || interestPrint) { printf("done history "); v.print(); }
    } else {
      Value v = tt.get(h, d - 1);
      // if (print) { h.print(); v.print(); }
      if (v.isCut<MAX>(beta)) {
        Value vv = v.relaxBound<MAX>();
        if (limitPrint || interestPrint) {
          printf("tt cut ");
          vv.print();
        }
        return vv;
      }
      if (v.isEnough(beta)) {
        if (limitPrint || interestPrint) { printf("done isEnough d %d max %d p %d ", d, (int)MAX, p); h.print(); v.print(); }
        acc = acc.accumulate<MAX>(v);
        SET(p, done);
      }
    }
  }
  
  stack[d] = n;
  if (d == 0) {
    if (limitPrint) {
      for (int i = LIMIT; i >= 0; --i) {
        stack[i].print();
        printf("%d %s ", (LIMIT - i), ((LIMIT - i) & 1) ? "MIN" : "MAX");
      }
      assert(false);
    }        
    int value = acc.getValue();
    assert(!MAX || value < beta);
    // acc = Value::makeUnknown(MAX ? value : -N);
    acc = Value::makeUnknown(beta - 1);
  }
  acc.updateHistoryPos(historyDepth);  
  if (d) {
    history->push(hash, d);  
    for (int i = 0; i < nMoves; ++i) {
      int p = moves[i];
      if (!IS(p, done)) {
        Hash h = hashes[i];
        Node sub = n.play<MAX ? BLACK : WHITE>(p);
        Value v = miniMax<!MAX>(sub, h, history, beta, d - 1);
        if (interestPrint) {
          sub.print();
          h.print();
          printf("%s ", MAX ? "MAX" : "MIN");
          v.print();
        }
        tt.set(h, v, d - 1);
        if (v.isCut<MAX>(beta)) {
          if (interestPrint) { printf("cut %d\n", d); }
          acc = v.relaxBound<MAX>();
          break;
        }
        acc = acc.accumulate<MAX>(v);
      }
    }
    history->pop(hash, d);
  }
  if (limitPrint) {
    for (int i = LIMIT; i >= d; --i) {
      stack[i].print();
      printf("* %d %s ", (LIMIT - i), ((LIMIT - i) & 1) ? "MIN" : "MAX");
    }
    printf("ret "); acc.print();
    assert(false);
  }
  return acc;
}

int main(int argc, char **argv) {
  Node n;
  // n.setup("x.xx.xxoxoooxo.o", 0);
  
  Driver driver;
  driver.mtd(n);
}

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

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
    auto it = map.find(hash.situationHash);
    return it == map.end() ? 0 : it->second;
  }
  
  bool push(const Hash &hash, int d) {
    bool added = map.emplace(hash.situationHash, d).second;
    return added;
  }

  void pop(const Hash &hash, int d) {
    auto n = map.erase(hash.situationHash);
    assert(n == 1);
    (void) n;
  }
};

void Driver::mtd(const Node &root, int iniDepth) {
  Hash hash;
  History history;
  int beta = N;
  int d = iniDepth;
  
  while (true) {
    Value v = miniMax<true>(root, hash, &history, beta, d);
    assert(v.isEnough(beta) || v.isDepthLimited());
    printf("MTD %d, beta %d: ", d, beta); v.print();
    if (v.isDepthLimited()) {
      ++d;
    } else {
      if (v.upp < beta) {
        beta = v.upp;
        if (beta == -N) { break; }        
      } else {
        assert(v.low == beta);
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
    n = ((i&1) == 0) ? n.play<true>(p) : n.play<false>(p);
    n.print();
  }
  */
}

template<bool MAX>
Value Driver::miniMax(const Node &n, const Hash &hash, History *history, const int beta, int d) {
  bool interest = false && d >= 10;
  assert(d >= 0);
  Value v = tt.get(hash, d, beta);
  if (v.isEnough(beta) /*|| v.isDepthLimited()*/) { return v; }
  if (v.isNone()) {
    v = n.score<MAX>(beta);
    tt.set(hash, v, d, beta);
    if (v.isEnough(beta)) {
      // tt.set(hash, v, d, beta);
      return v;
    }
  }
  assert(!n.isEnded());
  // assert(!v.isDepthLimited());
  Value acc = Value::makeExact(MAX ? -N : N);
  
  Vect<byte, N+1> moves;
  n.genMoves<MAX>(moves);
  int nMoves = moves.size();
  assert(nMoves > 0);
  Hash hashes[nMoves];
  uint64_t todo = 0;
  uint64_t todoLast = 0;
  
  for (int i = 0; i < nMoves; ++i) {
    int p = moves[i];
    Hash h = n.hashOnPlay<MAX>(hash, p);
    hashes[i] = h;
    int historyPos = (p == PASS) ? 0 : history->depthOf(h);
    if (historyPos) {
      assert(historyPos > d);
      acc.updateHistoryPos(historyPos);
    } else {
      Value v = tt.get(h, d - 1, beta);
      if (v.isCut<MAX>(beta)) {
        if (interest) {
          printf("%d hash cut: %d ", d, p); v.print(); n.print();
        }
        Value vv = v.relaxBound<MAX>();
        tt.set(hash, vv, d, beta);
        return vv;
      } else if (/*v.isDepthLimited() || */v.isCut<!MAX>(beta)) {
        acc = acc.accumulate<MAX>(v);
      } else if (v.isDepthLimited()) {
        SET(p, todo);
      } else {
        SET(p, todo);
      }
    }
  }
  
  if (todo | todoLast) {
    if (d == 0) {
      acc = Value::makeDepthLimited(acc.historyPos);
    } else {
      bool pushed = history->push(hash, d);
      for (int i = 0; i < nMoves; ++i) {
        int p = moves[i];
        if (IS(p, todo)) {
          Hash h = hashes[i];
          Node sub = n.play<MAX>(p);
          stack.push_back(p);
          Value v = miniMax<!MAX>(sub, h, history, beta, d - 1);
          stack.pop_back();
          assert(!v.isNone());
          if (v.isCut<MAX>(beta)) {
            if (interest) {
              printf("%d cut: %d ", d, p); v.print();
              for (int pp : stack) { printf("%d,", pp); }
              sub.print();
            }
            acc = v.relaxBound<MAX>();
            goto out;
          } else {
            acc = acc.accumulate<MAX>(v);
          }
        }
      }/*
      for (int i = 0; i < nMoves; ++i) {
        int p = moves[i];
        if (IS(p, todoLast)) {
          Hash h = hashes[i];
          Node sub = n.play<MAX>(p);
          Value v = miniMax<!MAX>(sub, h, history, beta, d - 1);
          assert(!v.isNone());
          if (v.isCut<MAX>(beta)) {
            acc = v.relaxBound<MAX>();
            goto out;
          } else {
            acc = acc.accumulate<MAX>(v);
          }
        }
        }*/

    out:
      if (pushed) { history->pop(hash, d); }
    }
  }
  tt.set(hash, acc, d, beta);
  return acc;
}

int main(int argc, char **argv) {
  int d;
  int depth = (argc >= 2 && (d = atoi(argv[1])) > 0) ? d : 16;
  Node n;
  // n.setup("xx.xoxxx.oo.o.oo", 0, P(2, 0));
  Driver driver;
  printf("SIZE %d, depth %d\n", SQ_SIZE, depth);
  driver.mtd(n, depth);
}

template<bool MAX>
int Driver::extract(const Node &n, const Hash &hash, History *history, const int beta, int d, int limit, std::vector<int> &moves) {
  assert(limit > 0);
  --limit;
  if (n.isEnded()) {
    assert(n.score<MAX>(beta).value == beta);
    minMoves = moves;
    return 0;
  }
  if (limit == 0 || d == 0) {
    return 1;
  }
  Vect<byte, N+1> subMoves;
  n.genMoves<MAX>(subMoves);
  int nMoves = subMoves.size();
  assert(nMoves > 0);
  history->push(hash, d);
  // int minP = 0;
  for (int i = 0; i < nMoves; ++i) {
    int p = subMoves[i];
    Hash h = n.hashOnPlay<MAX>(hash, p);
    if (!history->depthOf(h)) {
      Node sub = n.play<MAX>(p);
      Value v = miniMax<!MAX>(sub, h, history, beta, d - 1);
      // assert(!v.isCut<MAX>(beta));
      if (v.isEnough(beta) && v.low == beta) {
        moves.push_back(p);
        int subLimit = extract<!MAX>(sub, h, history, beta, d - 1, limit, moves);
        moves.pop_back();
        if (subLimit < limit) {
          limit = subLimit;
          if (limit == 0) { break; }
        }
      }
    }
  }
  history->pop(hash, d);
  return limit + 1;  
}

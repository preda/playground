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
  
  bool push(const Hash &hash, int d) {
    bool added = map.emplace(hash.hash, d).second;
    return added;
  }

  void pop(const Hash &hash, int d) {
    auto n = map.erase(hash.hash);
    assert(n == 1);
    (void) n;
  }
};

void Driver::mtd(const Node &root, int iniDepth) {
  Hash hash;
  History history;
  int beta = 3; //N;
  int d = iniDepth;

  while (true) {
    Value v = miniMax(root, hash, &history, beta, d);
    printf("MTD %d, beta %d: ", d, beta);
    v.print();
    
    assert(v.isEnough(beta) || v.isDepthLimited());

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
    break;
  }
}

Value Driver::miniMax(const Node &n, History *history, const int beta, const int d) {
  assert(d >= 0);
  Value v = tt.get(hash, d, beta);

  if (v.isNone()) {
    v = n.score(beta);
    tt.set(hash, v, d, beta);
    if (v.isEnough(beta)) {
      if(interest) { printf("score eval\n"); }
      return v;
    }
  }
  assert(!n.isEnded());
  // assert(!v.isDepthLimited());
  Value acc = Value::makeExact(-N);
  
  Vect<byte, N+1> moves;
  n.genMoves(moves);
  int nMoves = moves.size();
  assert(nMoves > 0);
  Hash hashes[nMoves];
  uint64_t todo = 0;
  uint64_t todoLast = 0;

  for (int i = 0; i < nMoves; ++i) {
    int p = moves[i];
    Hash h = n.hashOnPlay(hash, p);
    hashes[i] = h;
    int historyPos = /*(p == PASS) ? 0 :*/ history->depthOf(h);
    if (historyPos) {
      if (interest) { printf("%d hrej: %d (%d)\n", d, p, historyPos); }
      assert(historyPos > d);
      acc.updateHistoryPos(historyPos);
    } else {
      Value v = tt.get(h, d - 1, beta);
      if (v.isCut<true>(beta)) {
        if (interest) {
          printf("%d hc: %d ", d, p); v.print(); n.print();
        }
        Value vv = v.relaxBound();
        tt.set(hash, vv, d, beta);
        return vv;
      } else if (v.isCut<false>(beta)) {
        acc = acc.accumulate(v);
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
          Node sub = n.play(p).swapSides();
          stack.push_back(p);
          Value v = miniMax(sub, h, history, -(beta - 1), d - 1).swapSide();
          stack.pop_back();
          assert(!v.isNone());
          if (v.isCut<MAX>(beta)) {
            if (interest) {
              printf("%d c: %d ", d, p); v.print();
              for (int pp : stack) { printf("%d,", pp); }
              sub.print();
            }
            acc = v.relaxBound<MAX>();
            goto out;
          } else {
            if (interest) { printf("%d + : %d ", d, p); v.print(); } 
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
  if (interest) {
    printf("%d moves %d non-hash %d ", d, nMoves, size(todo)); acc.print();
    n.print();
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

  /*
  bool interest = n == iNode;
  if (n == iNode) {
    printf("%d : ", d);
    for (int p : stack) { printf("%d ", p); };
    printf("\n");
  }

  if ((rootD - d) <= 3 && stack.size() >= interestStack.size()) {
    for (int i = 0, n = interestStack.size(); i < n; ++i) {
      if (interestStack[i] != stack[i]) { goto nope; }
    }
    interest = true;
    // printf("* %d\n", d);
  }
  nope:
  */

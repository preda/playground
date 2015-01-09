// (c) Copyright 2014 Mihai Preda. All rights reserved.

#include "Driver.hpp"
#include <stdio.h>
#include <assert.h>

template<typename T> class Post {
private:
  T f;
public:
  Post(T f) : f(f) {}
  ~Post() { f(); }
};

void Driver::mtd() {
  Node root;
  int min = -N, max = N;
  int beta = N;
  int d = 8;
  while (min < max) {
    int g = MAX(root, beta, d);
    printf("MTDF %d: [%d, %d] beta %d: %d\n", d, min, max, beta, g);
    if (g == UNKNOWN) {
      // printf("MTDF depth %d: [%d, %d] beta %d: %d\n", d, min, max, beta, g);
      d += 2;
    } else if (g >= beta) {
      min = g;
      beta = g + 1;
    } else {
      max = g;
      beta = g;
    }
  }
}

int Driver::MAX(const Node &n, const int beta, int d) {
  uint128_t hash = n.getHash();
  int bound;
  bool exact;
  std::tie(bound, exact) = tt.get(hash, d);
  if (bound == TT_NOT_FOUND) {
    assert(!exact);
    int min;
    std::tie(min, bound) = n.score();
    exact = min == bound;
    tt.set(hash, d, bound, exact);
  }
  if (exact || bound == UNKNOWN || bound < beta) {
    return bound;
  }
  if (d <= 0) {
    tt.set(hash, d, UNKNOWN, false);
    return UNKNOWN;
  }
  Vect<byte, N> moves;
  n.genMoves<BLACK>(moves);
  int max = -N;
  for (int p : moves) {
    int s = MIN(n.play<BLACK>(p), beta, d - 1);
    if (s == UNKNOWN) {
      max = UNKNOWN;
    } else if (s >= beta) {
      tt.set(hash, d, s, true);
      return s;
    } else if (max != UNKNOWN && s > max) {
      max = s;
    }
  }
  tt.set(hash, d, max, false);
  return max;
}

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

int main(int argc, char **argv) {
  Driver driver;
  driver.mtd();
}

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

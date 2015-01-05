// Copyright (c) Mihai Preda 2013-2014

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

int Driver::mtdf(int f, int d) {
  Node root;
  int min = -N, max = N;
  int beta = f;
  int g;
  while (min < max) {
    g = AB<BLACK>(root, beta, d);
    printf("MTDF %d: [%d, %d] beta %d: %d\n", d, min, max, beta, g);
    if (g == UNKNOWN) {
      // printf("MTDF depth %d: [%d, %d] beta %d: %d\n", d, min, max, beta, g);
      break;
    }
    if (g >= beta) {
      min = g;
      beta = g + 1;
    } else {
      max = g;
      beta = g;
    }
  }
  return g;
}

inline int negaUnknown(int g) { return (g == UNKNOWN) ? UNKNOWN : -g; }

inline int maxUnknown(int a, int b) {
  return (a == UNKNOWN || b == UNKNOWN) ? UNKNOWN : std::max(a, b);
}

template<int C> int Driver::AB(const Node &n, int beta, int d) {
  assert(C == (d & 1));
  // n.print();
  uint128_t hash = n.getHash();

  ScoreBounds bounds = tt.lookup(hash);
  int min = bounds.min, max = bounds.max;
  if (min >= beta || max < beta) {
    // printf("%d beta %d Transposition [%d %d]\n", d, beta, min, max);
    return min >= beta ? min : max;
  }
  bounds = n.score<C>();
  min = std::max<int>(min, bounds.min);
  max = std::min<int>(max, bounds.max);
  // printf("%d %d score [%d %d]\n", d, beta, min, max);
  if (min >= beta || max < beta) {
    // n.print();
    // printf("%d beta %d Score [%d %d]\n", d, beta, min, max);
    tt.set(hash, min, max);
    return min >= beta ? min : max;
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

  if (d <= 0) {
    
    return UNKNOWN;
  }
  
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

int main(int argc, char **argv) {
  Driver driver;
  int d = 6;
  while (true) {
    int g = driver.mtdf(0, d);
    printf("At depth %d: %d\n", d, g);
    break;
    d += 2;
  }
}

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

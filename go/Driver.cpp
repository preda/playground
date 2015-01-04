#include "Driver.hpp"
#include <stdio.h>

template<typename T> class Post {
private:
  T f;
public:
  Post(T f) : f(f) {}
  ~Post() { f(); }
};

int Driver::mtdf(int g, int d) {
  Node root;
  int min = -N, max = N, beta = g;
  while (min < max) {
    int g = AB<BLACK>(root, beta, d);
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

int main(int argc, char **argv) {
  Driver driver;
  int d = 8;
  while (true) {
    int g = driver.mtdf(0, d);
    if (g != UNKNOWN) {
      printf("Score %d\n", g);
      break;
    }
    d += 2;
  }
}

inline int negaUnknown(int g) { return (g == UNKNOWN) ? UNKNOWN : -g; }

inline int maxUnknown(int a, int b) {
  return (a == UNKNOWN || b == UNKNOWN) ? UNKNOWN : std::max(a, b);
}

template<int C> int Driver::AB(const Node &n, int beta, int d) {
  uint128_t hash = n.getHash();

  ScoreBounds bounds = tt.lookup(hash);
  int min = bounds.min, max = bounds.max;
  if (min >= beta) { return min; }
  if (max  < beta) { return max; }
  bounds = n.score<C>();
  min = std::max<int>(min, bounds.min);
  max = std::min<int>(max, bounds.max);
  if (min >= beta || max < beta) {
    tt.set(hash, min, max);
    return min >= beta ? min : max;
  }
  
  Vect<byte, N> moves;
  n.genMoves<C>(moves);
  int subBeta = -(beta - 1);
  for (int p : moves) {
    int s = -tt.lookup(n.hashOnPlay<C>(p)).max;
    if (s >= beta) {
      tt.set(hash, s, max);
      return s;
    }
  }

  int g = -N;
  for (int p : moves) {
    int s = negaUnknown(AB<1-C>(n.play<C>(p), subBeta, d - 1));
    if (s >= beta) {
      tt.set(hash, s, max);
      return s;
    }
    g = maxUnknown(g, s);
  }
  if (g != UNKNOWN) {
    tt.set(hash, min, g);    
  }
  return g;
}

template int Driver::AB<BLACK>(const Node&, int, int);
template int Driver::AB<WHITE>(const Node&, int, int);

/*
inline int negaMax(int g, int score) {
  if (g == UNKNOWN || score == UNKNOWN) {
    return UNKNOWN;
  }
  return std::max(g, -score);
}
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


  /*
  if (!n.isKo()) {
    int maxScore = n.finalScore<C>();
    if ((g != UNKNOWN && maxScore > g) || maxScore >= beta) {
      g = PLAY(PASS);
    }
  }
  */

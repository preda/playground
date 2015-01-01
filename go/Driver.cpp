#include "Driver.hpp"

template<int C> int Driver::AB(Node *n, int beta, int d) {    
  ScoreBounds info = tt.lookup(n->getHash());
  int min = info.min, max = info.max;
  if (min >= beta) { return min; }
  if (max  < beta) { return max; }
  info = n->score<C>();
  min = std::max(min, (int) info.min);
  max = std::min(max, (int) info.max);
  if (min >= beta || max < beta) {
    tt.set(n->getHash(), min, max);
    return min >= beta ? min : max;
  }
  int g = -N;
  Vect<byte, N> moves;
  n->genMoves<C>(moves);
  for (int p : moves) {
    Node sub = n->play<C>(p);
    int subScore = -AB<1-C>(&sub, -beta + 1, d + 1);
    g = std::max(g, subScore);
    if (g >= beta) { break; }      
  }
  if (g >= beta) {
    min = g;
  } else {
    max = g;
  }
  tt.set(n->getHash(), min, max);
  return g;
}

template int Driver::AB<BLACK>(Node *, int, int);
template int Driver::AB<WHITE>(Node *, int, int);

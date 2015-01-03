#include "Driver.hpp"

inline int negaMax(int g, int score) {
  if (g == SCORE_UNKNOWN || score == SCORE_UNKNOWN) {
    return SCORE_UNKNOWN;
  }
  return std::max(g, -score);
}

#define PLAY(p) negaMax(g, AB<1-C>(n.play<C>(p), -(beta - 1), d + 1))

template<typename T>
class Post {
private:
  T f;
public:
  Post(T f) : f(f) {}
  ~Post() { f(); }
};

template<int C> int Driver::AB(const Node &n, int beta, int d) {
  uint128_t hash = n.getHash();
  bool newInHistory = history.insert(hash).second;
  if(!newInHistory && !n.lastWasPass()) {
    return SCORE_UNKNOWN;
  }
  auto f = [this, hash, newInHistory](){ if (newInHistory) { history.erase(hash); }};
  Post<decltype(f)> onReturn(f);
  
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
  
  int g = -N;
  Vect<byte, N> moves;
  n.genMoves<C>(moves);
  for (int p : moves) {
    g = PLAY(p);
    if (g >= beta) {
      tt.set(hash, g, max);
      return g;
    }
  }

  if (!n.isKo()) {
    int maxScore = n.finalScore<C>();
    if ((g != SCORE_UNKNOWN && maxScore > g) || maxScore >= beta) {
      g = PLAY(PASS);
    }
  }

  if (g != SCORE_UNKNOWN) {
    tt.set(hash, min, g);
  }
  return g;
}

template int Driver::AB<BLACK>(const Node&, int, int);
template int Driver::AB<WHITE>(const Node&, int, int);

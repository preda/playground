#pragma once
#include "data.hpp"
#include "go.h"

static inline int size(uint64_t bits) { return __builtin_popcount(bits); }

#define SET(p, bits) bits |= (1ull << (p))
#define IS(p, bits) ((bits) & (1ull << (p)))

class Board {
private:
  int mColorToPlay;
  uint64_t hash;
  
public:
  uint64_t borderOrStone[2] = {0};
  uint64_t stone[2] = {0};
  uint64_t empty = 0;
  uint64_t groups[MAX_GROUPS] = {0};
  byte gids[BIG_N] = {0};
  
  Board();
 
  void putBorder(int p) {
    SET(p, borderOrStone[BLACK]);
    SET(p, borderOrStone[WHITE]);
  }

  template<int C> bool is(int p) { return IS(p, stone[C]); }
  bool isEmpty(int p) { return IS(p, empty); }
  bool isBorder(int p) { return IS(p, borderOrStone[BLACK]) && IS(p, borderOrStone[WHITE]); }

  
  void update() {
    stone[BLACK] = borderOrStone[BLACK] & ~borderOrStone[WHITE];
    stone[WHITE] = borderOrStone[WHITE] & ~borderOrStone[BLACK];
    empty = ~borderOrStone[BLACK] & ~borderOrStone[WHITE];
  }

  int colorToPlay() { return mColorToPlay; }
  void swapColorToPlay();

  uint64_t *newGroup();
  int groupLibs(int gid) { return size(groups[gid] & empty); }

  template<int C> bool play(int p);
  
  template<int C> unsigned neibGroups(int p);
  template<int C> void updateGroup(int p, int gid);
  template<int C> void removeGroup(int gid);
  template<int C> unsigned bensonAlive(uint64_t *points);
  
  void print(uint64_t, uint64_t);
  char charForPos(int p);

private:
  template<int C> bool tryCapture(int p);
};

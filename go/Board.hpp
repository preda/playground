#pragma once
#include "data.hpp"
#include "go.hpp"

static inline int size(uint64_t bits) { return __builtin_popcountll(bits); }
static inline bool IS(int p, uint64_t bits) { return (bits >> p) & 1; }
// #define IS(p, bits) ((bits) & (1ull << (p)))
#define SET(p, bits) bits |= (1ull << (p))

class Board {
private:
  uint64_t empty = 0;
  uint64_t border = 0;
  uint64_t stone[2] = {0};
  uint64_t groups[MAX_GROUPS] = {0};
  uint64_t hash;
  byte gids[BIG_N] = {0};
  int mColorToPlay;
  int koPos;
  
public:  
  Board();
   
  template<int C> bool is(int p) { return IS(p, stone[C]); }
  bool isEmpty(int p)  { return IS(p, empty); }
  bool isBorder(int p) { return IS(p, border); }
  
  int colorToPlay() { return mColorToPlay; }
  void swapColorToPlay();

  template<int C> bool isSuicide(int p);
  template<int C> void play(int p);
  template<int C> uint64_t hashOnPlay(int p);
  
  template<int C> uint64_t bensonAlive();

  void changeSide();
  uint64_t fullHash();
  template<typename T> uint64_t transformedHash(T t);
  
  void print(uint64_t, uint64_t);

private:
  void updateEmpty() { empty = ~border & ~stone[BLACK] & ~stone[WHITE]; }
  int newGid();
  
  int libsOfGroup(uint64_t group) { return size(group & empty); }
  int libsOfGid(int gid) { return libsOfGroup(groups[gid]); }
  int libsOfGroupAtPos(int p) { return libsOfGid(gids[p]); }
  template<int C> int sizeOfGroup(uint64_t group) { return size(group & stone[C]); }
  template<int C> int sizeOfGid(int gid) { return sizeOfGroup<C>(groups[gid]); }
      
  template<int C> unsigned neibGroups(int p);
  template<int C> void updateGroupGids(uint64_t group, int gid);
  
  int groupColor(int gid);
  char charForPos(int p);  
};

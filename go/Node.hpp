#pragma once
#include "data.hpp"
#include "go.hpp"

static inline int size(uint64_t bits) { return __builtin_popcountll(bits); }
static inline bool IS(int p, uint64_t bits) { return (bits >> p) & 1; }

class Node {
private:
  uint128_t hash;
  uint64_t empty;
  uint64_t stone[2];
  uint64_t points[2];
  uint64_t groups[MAX_GROUPS];
  int koPos;
  byte gids[BIG_N];

public:  
  Node();
  // Node(const Node &other) { memcpy(this, &other, sizeof(Node)); }
     
  template<int C> bool is(int p) { return IS(p, stone[C]); }
  bool isEmpty(int p)  { return IS(p, empty); }
  bool isBorder(int p) { return IS(p, BORDER); }
  
  template<int C> bool isSuicide(int p);
  
  template<int C> Node play(int p) const {
    Node node(*this);
    node.playInt<C>(p);
    return node;
  }

  template<int C> uint64_t bensonAlive();

  void changeSide();
  template<int C> uint128_t hashOnPlay(int p);
  uint128_t getHash() { return hash; }
  template<typename T> uint128_t transformedHash(T t);

  template<int C> void genMoves(Vect<byte, N> &outMoves);
  template<int C> ScoreBounds score();
  
  void print(uint64_t, uint64_t);

private:
  template<int C> void playInt(int p);
  
  void updateEmpty() { empty = ~(stone[BLACK] | stone[WHITE]) & INSIDE; }
  int newGid();
  
  int libsOfGroup(uint64_t group) { return size(group & empty); }
  int libsOfGid(int gid) { return libsOfGroup(groups[gid]); }
  int libsOfGroupAtPos(int p) { return libsOfGid(gids[p]); }
  template<int C> int sizeOfGroup(uint64_t group) { return size(group & stone[C]); }
  template<int C> int sizeOfGid(int gid) { return sizeOfGroup<C>(groups[gid]); }
  template<int C> int sizeOfGroupAtPos(int p) { return sizeOfGid<C>(gids[p]); }
      
  template<int C> unsigned neibGroups(int p);
  template<int C> void updateGroupGids(uint64_t group, int gid);

  template<int C> int valueOfMove(int pos);

  uint64_t maybeMoves();
  template<typename T>
  bool isSymmetry(T t);
  
  int groupColor(int gid);
  char charForPos(int p); 
};

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
  int nPass;
  byte gids[BIG_N];

public:  
  Node();
  // Node(const Node &other) { memcpy(this, &other, sizeof(Node)); }
     
  template<int C> bool is(int p) const { return IS(p, stone[C]); }
  bool isEmpty(int p)  const { return IS(p, empty); }
  bool isBorder(int p) const { return IS(p, BORDER); }
  
  template<int C> bool isSuicide(int p) const { return valueOfMove<C>(p) < 0; }
  
  template<int C> Node play(int p) const {
    Node node(*this);
    node.playAndHash<C>(p);
    return node;
  }

  uint128_t getHash() const { return hash; }
  template<int C> uint128_t hashOnPlay(int p) const;
  
  template<int C> void genMoves(Vect<byte, N> &outMoves) const;
  template<int C> ScoreBounds score() const;
  template<int C> int finalScore() const;
  
  bool isKo() const { return koPos != 0; }
  bool lastWasPass() const { return nPass > 0; }
  
  void print() const;
  void setUp(const char *s);

private:
  template<int C> void playAndHash(int p);
  template<int C> uint64_t playInt(int p);
  
  
  void updateEmpty() { empty = ~(stone[BLACK] | stone[WHITE]) & INSIDE; }
  int newGid();
  
  int libsOfGroup(uint64_t group) const { return size(group & empty); }
  int libsOfGid(int gid) const { return libsOfGroup(groups[gid]); }
  int libsOfGroupAtPos(int p) const { return libsOfGid(gids[p]); }
  template<int C> int sizeOfGroup(uint64_t group) const { return size(group & stone[C]); }
  template<int C> int sizeOfGid(int gid) const { return sizeOfGroup<C>(groups[gid]); }
  template<int C> int sizeOfGroupAtPos(int p) const { return sizeOfGid<C>(gids[p]); }
      
  template<int C> unsigned neibGroups(int p) const;
  template<int C> void updateGroupGids(uint64_t group, int gid);

  template<int C> int valueOfMove(int pos) const;
  template<int C> uint64_t bensonAlive() const;
  void enclosedRegions(uint64_t *outEnclosed) const;

  uint64_t maybeMoves() const;
  template<typename T> bool isSymmetry(T t) const;
  
  int groupColor(int gid) const;
  char charForPos(int p) const;
  template<int C> bool hasEyeSpace(uint64_t area) const;  
};

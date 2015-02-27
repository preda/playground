// (c) Copyright 2014 Mihai Preda. All rights reserved.

#pragma once

#include "data.hpp"
#include "go.hpp"
#include <tuple>

class Hash;
class Value;

static inline int size(uint64_t bits) { return __builtin_popcountll(bits); }
static inline bool IS(int p, uint64_t bits) { return (bits >> p) & 1; }

class Node {
private:
  uint64_t empty;
  uint64_t stoneBlack, stoneWhite;
  uint64_t pointsBlack, pointsWhite;
  uint64_t groups[MAX_GROUPS];
  int nPass;
  byte gids[BIG_N];

public:
  Node();

  void setup(const char *board, int nPass = 0);

  template<bool BLACK> inline uint64_t stone() const { return BLACK ? stoneBlack : stoneWhite; }
  template<bool BLACK> bool is(int p) const { return IS(p, stone<BLACK>()); }
  bool isEmpty(int p)  const { return IS(p, empty); }
  bool isBorder(int p) const { return IS(p, BORDER); }
  
  template<bool BLACK> Node play(int p) const {
    Node n(*this);
    n.playInt<BLACK>(p);
    return n;
  }

  template<bool BLACK> Hash hashOnPlay(const Hash &h, int p) const;
  
  template<bool BLACK> void genMoves(Vect<byte, N+1> &outMoves) const;
  template<bool MAX> Value score(int beta) const;
  int finalScore() const;
  
  void print(const char *s = 0) const;
  bool isEnded() const { return nPass >= 2; }
  bool operator==(const Node &n) const { return stoneBlack == n.stoneBlack && stoneWhite == n.stoneWhite && nPass == n.nPass; }
  
  int getNPass() const { return nPass; }
  
private:
  template<bool BLACK> void playInt(int p);
  
  
  void updateEmpty() { empty = ~(stoneBlack | stoneWhite) & INSIDE; }
  int newGid();
  
  int libsOfGroup(uint64_t group) const { return size(group & empty); }
  int libsOfGid(int gid) const { return libsOfGroup(groups[gid]); }
  int libsOfGroupAtPos(int p) const { return libsOfGid(gids[p]); }
  template<bool BLACK> int sizeOfGroup(uint64_t group) const { return size(group & stone<BLACK>()); }
  template<bool BLACK> int sizeOfGid(int gid) const { return sizeOfGroup<BLACK>(groups[gid]); }
  template<bool BLACK> int sizeOfGroupAtPos(int p) const { return sizeOfGid<BLACK>(gids[p]); }
      
  template<bool BLACK> unsigned neibGroups(int p) const;
  template<bool BLACK> void updateGroupGids(uint64_t group, int gid);

  template<bool BLACK> int valueOfMove(int pos) const;
  template<bool BLACK> uint64_t bensonAlive() const;
  std::pair<uint64_t, uint64_t> enclosedRegions() const;

  uint64_t maybeMoves() const;
  template<typename T> bool isSymmetry(T t) const;
  
  // bool groupIsBlack(int gid) const;
  char charForPos(int p) const;
  template<bool BLACK> bool hasEyeSpace(uint64_t area) const;  
};

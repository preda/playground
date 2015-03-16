// (c) Copyright 2014 Mihai Preda. All rights reserved.

#pragma once

#include "data.hpp"
#include "go.hpp"
#include "Hash.hpp"

#include <tuple>
#include <vector>

class Value;

static inline int size(uint64_t bits) { return __builtin_popcountll(bits); }

static inline bool IS(int p, auto bits) { return (bits >> p) & 1; }

class Node {
private:
  // root state
  uint64_t stoneBlack, stoneWhite;
  int koPos;
  int nPass;
  bool swapped;

  // group info
  byte gids[BIG_N];
  std::vector<uint64_t> groups;  

  // derived
  uint64_t empty;
  Hash hash;
  
public:
  Node();
  
  void setup(const char *board, int nPass = 0, int koPos = 0);

  bool isBlack(int p) const { return IS(p, stoneBlack); }
  bool isWhite(int p) const { return IS(p, stoneWhite); }
  
  bool isEmpty(int p)  const { return IS(p, empty); }
  bool isBorder(int p) const { return IS(p, BORDER); }
  
  Node play(int p) const;
  
  void genMovesBlack(Vect<byte, N+1> &outMoves) const;
  template<bool MAX> Value score(int beta) const;
  int finalScore() const;
  
  bool isEnded() const { return nPass >= 2; }
  bool operator==(const Node &n) const {
    return stoneBlack == n.stoneBlack && stoneWhite == n.stoneWhite && nPass == n.nPass && koPos == n.koPos;
  }
  
  int getNPass() const { return nPass; }
  bool isKo() const { return koPos != 0; }
  bool lastWasPass() const { return nPass > 0; }  
  void print(const char *s = 0) const;
  
private:
  uint64_t pointsBlack();
  uint64_t pointsWhite();

  Node swapAndPlay(int p);
  void swapSidesInt();
  
  void updateEmpty() { empty = ~(stoneBlack | stoneWhite) & INSIDE; }
  int newGid();
  
  int libsOfGroup(uint64_t group) const { return size(group & empty); }
  int libsOfGid(int gid) const { return libsOfGroup(groups[gid]); }
  int libsOfGroupAtPos(int p) const { return libsOfGid(gids[p]); }
  int sizeOfGroupBlack(uint64_t group) { return size(group & stoneBlack; }
  
  template<bool BLACK> int sizeOfGroup(uint64_t group) const { return size(group & stone<BLACK>()); }
  template<bool BLACK> int sizeOfGid(int gid) const { return sizeOfGroup<BLACK>(groups[gid]); }
  template<bool BLACK> int sizeOfGroupAtPos(int p) const { return sizeOfGid<BLACK>(gids[p]); }
      
  unsigned neibGroupsBlack(int p) const;
  void setGroupGid(uint64_t group, int gid);

  template<bool BLACK> int valueOfMove(int pos) const;
  uint64_t bensonAliveBlack() const;
  bool hasEyeSpaceWhite(uint64_t area) const;
  std::pair<uint64_t, uint64_t> enclosedRegions() const;

  // uint64_t maybeMoves() const { return empty; }
  char charForPos(int p) const;
};

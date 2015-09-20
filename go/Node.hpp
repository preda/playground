// (c) Copyright 2014 Mihai Preda. All rights reserved.

#pragma once

#include "data.hpp"
#include "go.hpp"
#include "Hash.hpp"

#include <tuple>
#include <vector>

class Value;

class Node {
private:
  Hash hash;
  uint64_t stoneBlack, stoneWhite;
  uint64_t pBlack, pWhite;
  int koPos;
  int nPass;
  bool swapped;
  byte gids[N];

  
public:
  Node(u64 black, u64 white);
  
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
  uint64_t pointsBlack() {
    if (pBlack == -1) { pBlack = bensonAliveBlack(); }
    return pBlack;
  }
  
  uint64_t pointsWhite() {
    assert(pWhite != -1);
    return pWhite;
  }

  Node swapAndPlay(int p);
  void swapSidesInt();
  
  void updateEmpty() { empty = ~(stoneBlack | stoneWhite) & INSIDE; }
  int newGid();
  
  int libsOfGroup(uint64_t group) const { return size(group & empty); }
  int sizeOfGroupBlack(uint64_t group) { return size(group & stoneBlack; }

  // int libsOfGid(int gid) const { return libsOfGroup(groups[gid]); }
  // int libsOfGroupAtPos(int p) const { return libsOfGid(gids[p]); }
  
  // template<bool BLACK> int sizeOfGroup(uint64_t group) const { return size(group & stone<BLACK>()); }
  // template<bool BLACK> int sizeOfGid(int gid) const { return sizeOfGroup<BLACK>(groups[gid]); }
  // template<bool BLACK> int sizeOfGroupAtPos(int p) const { return sizeOfGid<BLACK>(gids[p]); }
      
  unsigned neibGroupsBlack(int p) const;
  void setGroupGid(uint64_t group, int gid);

  int valueOfMoveBlack(int pos) const;
  uint64_t bensonAliveBlack() const;
  bool hasEyeSpaceWhite(uint64_t area) const;
  std::pair<uint64_t, uint64_t> enclosedRegions() const;

  char charForPos(int p) const;
};

#pragma once
#include "data.h"
#include "go.h"

struct Cell {
public:
  Cell(): color(EMPTY), group(0) { }
  Cell(int color, int group) : color(color), group(group) { }
  unsigned color:2;
  unsigned group:6;
} __attribute__((packed));

struct Group {
  Group(): size(0), libs(0), pos(0) { }
  Group(int size, int libs, int pos) :
    size((byte) size),
    libs((byte) libs),
    pos((byte) pos)
  { }
  
  byte size;
  byte libs;
  byte pos;
};

class Board {
private:
  int stonesOnBoard;
  uint64_t hash;
  
public:
  Cell cells[BIG_N];
  Group groups[MAX_GROUPS];
  int colorToPlay;  

  Board();

  int nStonesOnBoard() { return stonesOnBoard; }
  int color(int p) const { return cells[p].color; }
  Group *group(int p) { return groups + cells[p].group; }
  Group *newGroup();
  int libs(int p) { return group(p)->libs; }
  int groupColor(const Group &g) { return color(g.pos); }
  
  bool play(int p, int color);
  bool play(int p);  
  
  void updateGroup(int p, int gid);
  void updateGroupLibs(int p);
  void removeGroup(int p);
  
  unsigned neibGroupsOfColor(int p, int col);
  void bensonAlive(int col, Bitset &points, unsigned *outAliveBits);
  
  void print(const Bitset &, const Bitset &);
};

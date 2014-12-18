#include "Board.h"
#include "data.h"
#include "go.h"
#include <assert.h>
// #include <stdio.h>

#include "zobrist.h"

/*
struct State {
  Board board;
  int ko;

  State() : board(), ko(0) { }
};
*/

Board::Board() : colorToPlay(BLACK) {
  for (int y = 0; y < SIZE_Y + 1; ++y) {
    cells[y * BIG_X].color = BROWN;
    cells[y * BIG_X + SIZE_X + 1].color = BROWN;
  }
  for (int x = 0; x < BIG_X; ++x) {
    cells[x].color = BROWN;
    cells[(SIZE_Y + 1) * BIG_X + x].color = BROWN;
  }
  hash = hashSideToPlay(colorToPlay);
}

#define STEP(p) if (!seen.testAndSet(p) && t(p)) { open.push(p); }

template<typename T>
Bitset walk(int p, T t) {
  Bitset seen;
  Vect<byte, N> open;
  STEP(p);
  while (!open.isEmpty()) {
    const int p = open.pop();
    STEP(p + 1);
    STEP(p - 1);
    STEP(p + DELTA);
    STEP(p - DELTA);
  }
  return seen;
}
 
#undef STEP

void Board::updateGroupLibs(int p) {
  int libs = 0;
  int col = color(p);
  Cell *cells = this->cells;
  walk(p, [&libs, cells, col](int p) {
      int c = cells[p].color;
      if (c == col) { return true; }
      if (c == EMPTY) { ++libs; }
      return false;
    });
  group(p)->libs = libs;
}

void Board::updateGroup(int p, int gid) {
  int col = color(p);
  assert(isBlackOrWhite(col));  
  Cell cell(col, gid);
  int libs = 0;
  int size = 0;
  Cell *cells = this->cells;
  walk(p, [&libs, &size, cells, col, cell](int p) {
      int c = cells[p].color;
      if (c == col) { cells[p] = cell; ++size; return true; }
      if (c == EMPTY) { ++libs; }
      return false;
    });
  Group *g = groups + gid;
  g->size = size;
  g->libs = libs;
  g->pos = p;
}

void Board::removeGroup(int p) {
  int col = color(p);
  assert(isBlackOrWhite(col));
  Cell empty = Cell(EMPTY, 0);
  Cell *cells = this->cells;
  walk(p, [cells, col](int p) {
      if (cells[p].color == col) { cells[p] = Cell(EMPTY, 0); return true; }
      return false;
    });
}

Group *Board::newGroup() {
  for (Group *g = groups, *end = groups + MAX_GROUPS; g < end; ++g) {
    if (g->size == 0) { return g; }
  }
  assert(false && "max groups exceeded");
}

bool Board::play(int pos, int col) {
  assert(color(pos) == EMPTY);
  assert(isBlackOrWhite(col));
  int otherCol = 1 - col;
  int neighb[] = {pos+1, pos-1, pos+BIG_X, pos-BIG_X};
  bool captured = false;
  for (int p : neighb) {
    if (color(p) == otherCol && libs(p) == 1) {
      group(p)->size = 0;
      removeGroup(p);
      captured = true;
    }
  }
  if (captured) {
    for (Group *g = groups, *end = groups + MAX_GROUPS; g < end; ++g) {
      if (g->size > 0 && groupColor(*g) == col) {
        updateGroupLibs(g->pos);
      }
    }
  } else {
    bool suicide = true;
    for (int p : neighb) {
      if (color(p) == EMPTY || (color(p) == col && group(p)->libs > 1)) {
        suicide = false;
      }
    }
    if (suicide) { return false; }
  }
  bool needUpdateGroup = false;
  Group *g = 0;
  for (int p : neighb) {
    if (color(p) == col) {
      if (!g) {
        g = group(p);
      } else if (group(p) != g) {
        group(p)->size = 0;
        needUpdateGroup = true;
      }
    } else if (color(p) == otherCol) {
      --group(p)->libs;
    }
  }
  if (!g) { g = newGroup(); }
  int gid = g - groups;
  cells[pos] = Cell(col, gid);
  if (needUpdateGroup) { updateGroup(pos, gid); }
  return true;
}

unsigned Board::neibGroupsOfColor(int p, int col) {
  unsigned bits = 0;
  Cell c;
  c = cells[p+1]; if (c.color == col) { bits |= (1 << c.group); }
  c = cells[p-1]; if (c.color == col) { bits |= (1 << c.group); }
  c = cells[p+DELTA]; if (c.color == col) { bits |= (1 << c.group); }
  c = cells[p-DELTA]; if (c.color == col) { bits |= (1 << c.group); }
  return bits;
}

struct Region {
  unsigned border;
  Vect<byte, 4> vital;
  Bitset area;

  Region(): border(0), vital() { }
  Region(unsigned vitalBits, unsigned border, Bitset area) :
    border(border), area(area) {
    int i = 0;
    while (vitalBits) {
      if (vitalBits & 1) { vital.push(i); }
      ++i;
      vitalBits >>= 1;
    }
  }

  bool isCoveredBy(unsigned gidBits) {
    return (border & gidBits) == border;
  }

  /*
  void print() {
    printf("region border %x area %d vital ", border, area.size());
    for (int gid : vital) {
      printf("%d ", gid);
    }
    printf("\n");
  }
  */

  int size() { return area.size(); }
};

void Board::bensonAlive(int col, Bitset &points, unsigned *outAliveGids) {
  assert(isBlackOrWhite(col));
  int otherCol = 1 - col;
  Vect<Region, MAX_GROUPS> regions;
  Bitset seen;
  
  for (int y = 0; y < SIZE_Y; ++y) {
    for (int p = pos(y, 0), end = p + SIZE_X; p < end; ++p) {
      if (!seen.test(p) && color(p) == EMPTY) {
        unsigned vital = -1;
        unsigned border = 0;
        Bitset area;
        seen |= walk(p, [&area, &vital, &border, this, col](int p) {
            int c = color(p);
            if (c == EMPTY || c == (1-col)) {
              unsigned bits = neibGroupsOfColor(p, col);
              border |= bits;
              if (c == EMPTY) { vital &= bits; }
              area.set(p);
              return true;
            } else {
              return false;
            }
          });
        Region r = Region(vital, border, area);
        regions.push(r);
      }
    }
  }

  Vect<byte, MAX_GROUPS> aliveGids;
  unsigned aliveBits = 0;
  while (true) {
    int vitality[MAX_GROUPS] = {0};
    aliveGids.clear();
    for (Region r : regions) {
      for (int g : r.vital) {
        if (++vitality[g] >= 2) {
          aliveGids.push(g);
        }
      }
    }
    aliveBits = 0;
    if (aliveGids.isEmpty()) { break; }
    for (int g : aliveGids) {
      aliveBits |= (1 << g);
    }
    bool changed = false;
    for (Region &r : regions) {
      if (!r.vital.isEmpty() && !r.isCoveredBy(aliveBits)) {
        r.vital.clear();
        changed = true;
      }
    }
    if (!changed) { break; }
  }

  points.clear();
  *outAliveGids = aliveBits;
  if (!aliveGids.isEmpty()) {
    for (Region &r : regions) {
      if (r.isCoveredBy(aliveBits) && r.size() <= 8) {
        points |= r.area;
      }
    }
    for (int gid : aliveGids) {
      walk(groups[gid].pos, [this, &points, col](int p) {
          if (color(p) == col) {
            points.testAndSet(p);
            return true;
          } else {
            return false;
          }
        });
    }
  }
}

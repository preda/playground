#include "Board.hpp"
#include "data.hpp"
#include "go.h"

#include <initializer_list>
#include <assert.h>
#include <stdio.h>
#include "zobrist.h"

Board::Board() : mColorToPlay(BLACK), hash(0) {
  for (int y = 0; y < SIZE_Y; ++y) {
    putBorder(pos(y, SIZE_X));
    printf("%d ", pos(y, SIZE_X));
  }
  for (int x = 0; x < BIG_X; ++x) {
    printf("+%d %d ", pos(-1, x), pos(SIZE_Y, x));
    putBorder(pos(-1, x));
    putBorder(pos(SIZE_Y, x));
  }
  update();
  // printf("%Lx %Lx %Lx %Lx %Lx\n", borderOrStone[0], borderOrStone[1], stone[0], stone[1], empty);
}

void Board::swapColorToPlay() {
  mColorToPlay = 1 - mColorToPlay;
  hash ^= hashChangeSide();
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
/*
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
*/

template <int C> void Board::updateGroup(int p, int gid) {
  walk(p, [this, gid](int p) {
      bool isColor = is<C>(p);
      if (isColor) { gids[p] = gid; }
      return isColor;
    });
}

template<int C> void Board::removeGroup(int gid) {
  borderOrStone[C] &= ~(groups[gid] & stone[C]);
  update();
}


uint64_t *Board::newGroup() {
  for (uint64_t *g = groups, *end = groups + MAX_GROUPS; g < end; ++g) {
    if (*g == 0) { return g; }
  }
  assert(false && "max groups exceeded");
  return 0;
}

/*
bool Board::play(int pos) {
  int col = colorToPlay();
  play(pos, col);
  swapColorToPlay();
}
*/

#define NEIB(p) {p+1, p-1, p+DELTA, p-DELTA}

inline uint64_t shadow(int p) {
  return ((uint64_t)(1 | (7 << (BIG_X-1)) | (1 << (BIG_X + BIG_X)))) << (p - BIG_X);
}

template<int C> bool Board::tryCapture(int p) {
  if (is<C>(p) && groupLibs(gids[p]) == 1) {
    removeGroup<C>(p);
    groups[gids[p]] = 0;
    return true;
  } else {
    return false;
  }
}



template<int C> bool Board::play(int pos) {
  assert(isEmpty(pos));
  assert(isBlackOrWhite(C));
  bool captured = false;
  captured |= tryCapture<1-C>(pos+1);
  captured |= tryCapture<1-C>(pos-1);
  captured |= tryCapture<1-C>(pos+DELTA);
  captured |= tryCapture<1-C>(pos-DELTA);

  /*
  if (!captured) {
    bool suicide = true;
    for (int p : neighb) {
      if (color(p) == EMPTY || (color(p) == col && group(p)->libs > 1)) {
        suicide = false;
      }
    }
    if (suicide) { return false; }
  }
  */
  uint64_t *g = 0;
  bool needUpdate = false;
  for (int p : NEIB(pos)) {
    if (is<C>(p)) { 
      uint64_t *gg = groups + gids[p];
      if (!g) {
        g = gg;
      } else if (g != gg) {
        *g |= *gg;
        *gg = 0;
        needUpdate = true;
      }
    }
  }
  if (!g) { g = newGroup(); }
  int gid = g - groups;
  *g |= shadow(pos);
  SET(pos, borderOrStone[C]);
  update();
  if (needUpdate) {
    updateGroup<C>(pos, gid);
  } else {
    gids[pos] = gid;
  }
  return true;
}

template<int C> unsigned Board::neibGroups(int p) {
  unsigned bits = 0;
  for (int pp : NEIB(p)) {
    if (is<C>(pp)) { SET(gids[pp], bits); }
  }
  return bits;
}

struct Region {
  unsigned border;
  Vect<byte, 4> vital;
  uint64_t area;

  Region(): border(0), vital() { }
  Region(unsigned vitalBits, unsigned border, uint64_t area) :
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

  int size() { return ::size(area); }
};

template<int C> unsigned Board::bensonAlive(uint64_t *outPoints) {
  assert(isBlackOrWhite(C));
  Vect<Region, MAX_GROUPS> regions;
  Bitset seen;
  
  for (int y = 0; y < SIZE_Y; ++y) {
    for (int p = pos(y, 0), end = p + SIZE_X; p < end; ++p) {
      if (!seen[p] && isEmpty(p)) {
        unsigned vital = -1;
        unsigned border = 0;
        uint64_t area;
        uint64_t emptyOrOth = empty | stone[1 - C];
        seen |= walk(p, [&area, &vital, &border, this](int p) {
            if (IS(p, borderOrStone[C])) {
              return false;
            }
            unsigned gidBits = neibGroups<C>(p);
            border |= gidBits;
            SET(p, area);
            if (isEmpty(p)) {
              vital &= gidBits;
            }
            return true;
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

  uint64_t points = 0;
  if (!aliveGids.isEmpty()) {
    for (Region &r : regions) {
      if (r.isCoveredBy(aliveBits) && r.size() <= 8) {
        points |= r.area;
      }
    }
    for (int gid : aliveGids) {
      points |= groups[gid] & stone[C];
    }
  }
  *outPoints = points;
  return aliveBits;
}

template bool Board::play<BLACK>(int);
template bool Board::play<WHITE>(int);
template unsigned Board::bensonAlive<BLACK>(uint64_t *points);
template unsigned Board::bensonAlive<WHITE>(uint64_t *points);

#include "Board.hpp"
#include "data.hpp"
#include "go.h"

#include <initializer_list>
#include <assert.h>
#include <stdio.h>
#include "zobrist.h"

Board::Board() : mColorToPlay(BLACK), hash(0) {
  for (int y = 0; y < SIZE_Y; ++y) {
    SET(pos(y, SIZE_X), border);
  }
  for (int x = 0; x < BIG_X; ++x) {
    SET(pos(-1, x), border);
    SET(pos(SIZE_Y, x), border);
  }
  updateEmpty();
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

template <int C> void Board::updateGroupGids(uint64_t group, int gid) {
  uint64_t inside = group & stone[C];
  while (inside) {
    int p = __builtin_ctzll(inside);
    inside &= inside - 1; // ~(1ull << p);
    gids[p] = gid;
  }
}

int Board::newGid() {
  for (uint64_t *g = groups, *end = groups + MAX_GROUPS; g < end; ++g) {
    if (*g == 0) { return g - groups; }
  }
  assert(false && "max groups exceeded");
  return -1;
}

#define NEIB(p) {p+1, p-1, p+DELTA, p-DELTA}

inline uint64_t shadow(int p) {
  return ((uint64_t)(1 | (7 << (BIG_X-1)) | (1 << (BIG_X + BIG_X)))) << (p - BIG_X);
}

template<int C> bool Board::isSuicide(int pos) {
  for (int p : NEIB(pos)) {
    if (isEmpty(p)) {
      return false;
    } else if (is<C>(p)) {
      if (libsOfGroupAtPos(p) > 1) {
        return false;
      }
    } else if (is<1-C>(p) && libsOfGroupAtPos(p) == 1) {
      return false;
    }
  }
  return true;
}

template<int C> void Board::play(int pos) {
  assert(isEmpty(pos));
  assert(isBlackOrWhite(C));
  
  uint64_t group = shadow(pos);
  
  int newGid = -1;
  bool isSimple = true;
  for (int p : NEIB(pos)) {
    if (isEmpty(p)) { continue; }
    
    int gid = gids[p];
    if (is<C>(p)) {
      group |= groups[gid];
      if (newGid == -1) {
        newGid = gid;
      } else if (newGid != gid) {
        isSimple = false;
        groups[gid] = 0;
      }
    } else if (is<1-C>(p) && libsOfGid(gid) == 1) {
        stone[1-C] &= ~groups[gid];
        groups[gid] = 0;
    }
  }
  SET(pos, stone[C]);
  updateEmpty();  
  if (newGid == -1) { newGid = this->newGid(); }
  groups[newGid] = group;
  if (isSimple) {
    gids[pos] = newGid;
  } else {
    updateGroupGids<C>(group, newGid);
  }
  assert(libsOfGroupAtPos(pos) > 0);
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
        uint64_t borderOrCol = border | stone[C];
        seen |= walk(p, [&area, &vital, &border, borderOrCol, this](int p) {
            if (IS(p, borderOrCol)) { return false; }
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

template void Board::play<BLACK>(int);
template void Board::play<WHITE>(int);
template bool Board::isSuicide<BLACK>(int);
template bool Board::isSuicide<WHITE>(int);
template unsigned Board::bensonAlive<BLACK>(uint64_t *points);
template unsigned Board::bensonAlive<WHITE>(uint64_t *points);

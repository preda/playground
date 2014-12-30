#include "Node.hpp"
#include "data.hpp"
#include "zobrist.hpp"
#include "go.hpp"

#include <initializer_list>
#include <assert.h>
#include <stdio.h>

Node::Node() {
  for (int y = 0; y < SIZE_Y; ++y) {
    SET(P(y, -1), border);
    SET(P(y, SIZE_X), border);
  }
  for (int x = 0; x < BIG_X; ++x) {
    SET(P(-1, x), border);
    SET(P(SIZE_Y, x), border);
  }
  updateEmpty();
}

template <int C> void Node::updateGroupGids(uint64_t group, int gid) {
  for (int p : Bits(group & stone[C])) { gids[p] = gid; }
}

int Node::newGid() {
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

template<int C> bool Node::isSuicide(int pos) {
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

template<typename T>
uint128_t Node::transformedHash(T t) {
  uint128_t hash = 0;
  for (int p : Bits(stone[BLACK])) { hash ^= hashPos<BLACK>(t(p)); }
  for (int p : Bits(stone[WHITE])) { hash ^= hashPos<WHITE>(t(p)); }
  if (koPos) { hash ^= hashKo(t(koPos)); }
  return hash;
}

// uint128_t Node::fullHash() { return transformedHash([](int p) { return p; }); }

void Node::changeSide() {
  hash ^= hashSide();
}

template<int C> uint128_t Node::hashOnPlay(int pos) {
  uint64_t capture = 0;
  bool maybeKo = true;
  for (int p : NEIB(pos)) {
    if (isEmpty(p) || is<C>(p)) {
      maybeKo = false;
    } else if (is<1-C>(p)) {
      int gid = gids[p];
      uint64_t g = groups[gid];
      if (libsOfGroup(g) == 1) {
        capture |= g;
      }
    }
  }
  bool isKo = maybeKo && sizeOfGroup<1-C>(capture) == 1;
  uint128_t newHash = hash;
  assert(!isKo || koPos != pos);
  if (koPos) { newHash ^= hashKo(koPos); }
  if (isKo) { newHash ^= hashKo(pos); }
  if (capture) { for (int p : Bits(capture & stone[1-C])) { newHash ^= hashPos<1-C>(p); } }
  return newHash;
}

template<int C> void Node::play(int pos) {
  assert(isEmpty(pos));
  assert(isBlackOrWhite(C));
  
  uint64_t group = shadow(pos);
  
  int newGid = -1;
  bool isSimple = true;
  bool maybeKo = true;
  uint64_t capture = 0;
  for (int p : NEIB(pos)) {
    if (isEmpty(p)) {
      maybeKo = false;
      continue;
    }    
    int gid = gids[p];
    if (is<C>(p)) {
      maybeKo = false;
      group |= groups[gid];
      if (newGid == -1) {
        newGid = gid;
      } else if (newGid != gid) {
        isSimple = false;
        groups[gid] = 0;
      }
    } else if (is<1-C>(p)) {
      uint64_t g = groups[gid];
      if (libsOfGroup(g) == 1) {
        capture |= g;
        groups[gid] = 0;
      }
    }
  }
  bool isKo = maybeKo && sizeOfGroup<1-C>(capture) == 1;
  koPos = isKo ? pos : 0;
  stone[1-C] &= ~capture;
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

template<int C> unsigned Node::neibGroups(int p) {
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

template<typename T>
Bitset walk(int p, T t) {
#define STEP(p) if (!seen.testAndSet(p) && t(p)) { open.push(p); }
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
#undef STEP
}

template<int C> uint64_t Node::bensonAlive() {
  // assert(isBlackOrWhite(C));
  Vect<Region, MAX_GROUPS> regions;
  Bitset seen;
  uint64_t borderOrCol = border | stone[C];
  for (int y = 0; y < SIZE_Y; ++y) {
    for (int p = P(y, 0), end = p + SIZE_X; p < end; ++p) {
      if (!seen[p] && isEmpty(p)) {
        unsigned vital = -1;
        unsigned border = 0;
        uint64_t area;

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
    uint64_t stonePoints = 0;
    for (int gid : aliveGids) {
      stonePoints |= groups[gid];
    }
    points |= (stonePoints & stone[C]);
  }
  return points;
}

template void Node::play<BLACK>(int);
template void Node::play<WHITE>(int);
template bool Node::isSuicide<BLACK>(int);
template bool Node::isSuicide<WHITE>(int);
template uint64_t Node::bensonAlive<BLACK>();
template uint64_t Node::bensonAlive<WHITE>();





/*  
  uint64_t hash = 0;
  for (int y = 0; y < SIZE_Y; ++y) {
    int p = pos(y, 0);
    for (int x = 0; x < SIZE_X; ++x, ++p) {
      if (is<BLACK>(p)) {
        hash ^= hashPos<BLACK>(p);
      } else if (is<WHITE>(p)) {
        hash ^= hashPos<WHITE>(p);
      }
    }
  }
  if (koPos) {
    hash ^= hashKo(koPos);
  }
  if (colorToPlay() == WHITE) {
    hash ^= hashWhiteToPlay();
  }
  return hash;
}
*/

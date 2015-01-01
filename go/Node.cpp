#include "Node.hpp"
#include "data.hpp"
#include "zobrist.hpp"
#include "go.hpp"

#include <initializer_list>
#include <algorithm>
#include <assert.h>
#include <stdio.h>

Node::Node() :
  hash(0),
  empty(INSIDE),
  stone{0},
  points{0},
  groups{0},
  koPos(0),
  gids{0}
{ }

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

#define NEIB(p) {p+1, p+DELTA, p-1, p-DELTA}

inline uint64_t shadow(int p) {
  return ((uint64_t)(1 | (7 << (BIG_X-1)) | (1 << (BIG_X + BIG_X)))) << (p - BIG_X);
}

template<int C> int Node::valueOfMove(int pos) {
  bool isSuicide = true;
  int value = 0;
  int prevGid = -1;
  for (int p : NEIB(pos)) {
    if (isEmpty(p)) {
      isSuicide = false;
      ++value;
    } else if (isBorder(p)) {
      continue;
    } else {
      int gid = gids[p];
      uint64_t group = groups[gids[p]];
      int libs = libsOfGroup(group);      
      if (is<C>(p)) {
        if (libs > 1) { isSuicide = false; }
        if (libs == 1) {
          value += 2;
        } else if (libs == 2) {
          ++value;
        }
        if (gid != prevGid) {
          prevGid = gid;
          ++value;
        }
      } else if (is<1-C>(p)) {
        if (libs == 1) {
          isSuicide = false;
          value += sizeOfGroup<1-C>(group) + 1;
        } else if (libs == 2) {
          ++value;
        }
      }
    }
  }
  return isSuicide ? -1 : value;
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

template<int C> void Node::playInt(int pos) {
  static_assert(isBlackOrWhite(C), "color");
  assert(isEmpty(pos));  
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
  points[C] = bensonAlive<C>();
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
uint64_t walk(int p, T t) {
#define STEP(p) if (!IS(p, seen)) { SET(p, seen); if (t(p)) { open.push(p); }}
  uint64_t seen = 0;
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

void Node::enclosedRegions(uint64_t *outEnclosed) {
  uint64_t emptyNotSeen = empty;
  uint64_t enclosedBlack = 0;
  uint64_t enclosedWhite = 0;
  while (emptyNotSeen) {
    int p = firstOf(emptyNotSeen);
    uint64_t area = 0;
    bool touchesBlack = false;
    bool touchesWhite = false;
    uint64_t seen = walk(p, [&area, &touchesBlack, &touchesWhite, this](int p) {
        if (isEmpty(p)) {
          SET(p, area);
          return true;
        } else if (is<BLACK>(p)) {
          touchesBlack = true;
        } else if (is<WHITE>(p)) {
          touchesWhite = true;
        }
        return false;
      });
    emptyNotSeen &= ~seen;
    assert(touchesBlack || touchesWhite);
    if (touchesBlack && ~touchesWhite) {
      enclosedBlack |= area;
    } else if (touchesWhite && !touchesBlack) {
      enclosedWhite |= area;
    }
  }
  outEnclosed[0] = enclosedBlack;
  outEnclosed[1] = enclosedWhite;
}

template<int C> uint64_t Node::bensonAlive() {
  Vect<Region, MAX_GROUPS> regions;
  uint64_t borderOrCol = BORDER | stone[C];
  uint64_t emptyNotSeen = empty;

  while (emptyNotSeen) {
    int p = firstOf(emptyNotSeen);
    unsigned vital = -1;
    unsigned border = 0;
    uint64_t area = 0;
    uint64_t seen = walk(p, [&area, &vital, &border, borderOrCol, this](int p) {
        if (IS(p, borderOrCol)) { return false; }
        unsigned gidBits = neibGroups<C>(p);
        border |= gidBits;
        SET(p, area);
        if (isEmpty(p)) {
          vital &= gidBits;
        }
        return true;
      });
    emptyNotSeen &= ~seen;
    Region r = Region(vital, border, area);
    regions.push(r);
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

template<typename T> static uint64_t transform(uint64_t points, T t) {
  uint64_t r = 0;
  for (int p : Bits(points)) { SET(t(p), r); }
  return r;
}

uint64_t reflectX(uint64_t points) {
  return transform(points, [](int p) { return P(Y(p), SIZE_X - 1 - X(p)); });
}

uint64_t reflectY(uint64_t points) {
  return transform(points, [](int p) { return P(SIZE_Y - 1 - Y(p), X(p)); });
}

uint64_t reflectDiag(uint64_t points) {
  return transform(points, [](int p) { return P(X(p), Y(p)); });
}

template<typename T>
bool Node::isSymmetry(T t) {
  return stone[BLACK] == t(stone[BLACK]) && stone[WHITE] == t(stone[WHITE]);
}

uint64_t Node::maybeMoves() {
  uint64_t area = INSIDE;
  if (!koPos && size(empty) > N - 10) {
    if (isSymmetry(reflectX))    { area &= HALF_X; }
    if (isSymmetry(reflectY))    { area &= HALF_Y; }
    if (isSymmetry(reflectDiag)) { area &= HALF_DIAG; }
  }
  return area;
}

template<int C> void Node::genMoves(Vect<byte, N> &moves) {
  int tmp[N];
  int n = 0;
  uint64_t area = maybeMoves();
  for (int p : Bits(area)) {
    int v = valueOfMove<C>(p);
    if (v >= 0) {
      tmp[n++] = ((1000 - v) << 8) | p;
    }
  }
  std::sort(tmp, tmp + n);
  moves.clear();
  for (int *pt = tmp, *end = tmp + n; pt < end; ++pt) {
    moves.push(*pt & 0xff);
  }
}

template<int C> ScoreBounds Node::score() {
  if (nPass < 3) {
    return {(signed char) (-N + 2 * size(points[C])), (signed char) (N - 2 * size(points[1-C]))};
  } else {
    uint64_t enclosed[2] = {0};
    enclosedRegions(enclosed);
    uint64_t total[2];
    for (int i : {BLACK, WHITE}) {
      total[i] = points[i] | stone[i] | enclosed[i];
    }
    assert((total[BLACK] & total[WHITE]) == 0);
    int score = size(total[C]) - size(total[1-C]);
    return {(signed char)score, (signed char)score};
  }
}

template void Node::playInt<BLACK>(int);
template void Node::playInt<WHITE>(int);
template uint64_t Node::bensonAlive<BLACK>();
template uint64_t Node::bensonAlive<WHITE>();
template ScoreBounds Node::score<BLACK>();
template ScoreBounds Node::score<WHITE>();
template void Node::genMoves<BLACK>(Vect<byte, N> &);
template void Node::genMoves<WHITE>(Vect<byte, N> &);
template int Node::valueOfMove<BLACK>(int);
template int Node::valueOfMove<WHITE>(int);

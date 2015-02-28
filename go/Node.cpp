// (c) Copyright 2014 Mihai Preda. All rights reserved.

#include "Node.hpp"
#include "data.hpp"
#include "Hash.hpp"
#include "Value.hpp"
#include "go.hpp"

#include <initializer_list>
#include <algorithm>
#include <assert.h>
#include <stdio.h>

Node::Node() :
  empty(INSIDE),
  stoneBlack(0),
  stoneWhite(0),
  pointsBlack(0),
  pointsWhite(0),
  groups{0},
  koPos(0),
  nPass(0),
  gids{0}
{ }

void Node::setup(const char *board, int nPass, int koPos) {
  for (int y = 0; y < SIZE_Y; ++y) {
    for (int x = 0; x < SIZE_X; ++x) {
      char c = *board++;
      assert(c == 'x' || c == 'o' || c == '.');
      int p = P(y, x);
      if (c == 'x') {
        playInt<true>(p);
      } else if (c == 'o') {
        playInt<false>(p);
      }
    }
  }
  assert(!*board);
  this->nPass = nPass;
  this->koPos = koPos;
}

template <bool BLACK> void Node::updateGroupGids(uint64_t group, int gid) {
  for (int p : Bits(group & stone<BLACK>())) { gids[p] = gid; }
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

template<bool BLACK> int Node::valueOfMove(int pos) const {
  assert(isEmpty(pos) && koPos != pos);
  bool isSelfEye = true;
  int value = 0;
  uint64_t mergeGroup = shadow(pos);
  uint64_t capture = 0;
  for (int p : NEIB(pos)) {
    if (isEmpty(p)) {
      isSelfEye = false;
    } else if (isBorder(p)) {
      continue;
    } else {
      uint64_t group = groups[gids[p]];
      if (is<BLACK>(p)) {
        mergeGroup |= group;
        if (isSelfEye && libsOfGroup(group) == 1) {
          isSelfEye = false;
        }
      } else {
        assert(is<!BLACK>(p));
        isSelfEye = false;
        int libs = libsOfGroup(group);        
        if (libs == 1) {
          capture |= group;
        } else if (libs == 2) {
          value += 10;
        } else {
          value += 3;
        }
      }
    }
  }
  if (isSelfEye) { return -1; }
  capture &= stone<!BLACK>();
  uint64_t newEmpty = (empty & ~(1ull << pos)) | capture;
  int mergeLibs = size(mergeGroup & newEmpty);
  if (mergeLibs == 0) { return -1; } // suicide
  return value + (size(capture) + mergeLibs - 1) * 10;
}

template<bool BLACK> Hash Node::hashOnPlay(const Hash &hash, int pos) const {
  uint64_t capture = 0;
  bool isKo = false;
  int newNPass = 0;
  if (pos == PASS) {
    if (koPos) {
      assert(nPass == 0);
    } else {
      assert(nPass < 2);
      newNPass = nPass + 1;
    }
  } else {
    assert(isEmpty(pos));
    assert(pos != koPos);
    newNPass = 0;
    bool maybeKo = true;
    for (int p : NEIB(pos)) {
      if (isEmpty(p) || is<BLACK>(p)) {
        maybeKo = false;
      } else if (is<!BLACK>(p)) {
        int gid = gids[p];
        uint64_t g = groups[gid];
        if (libsOfGroup(g) == 1) {
          capture |= g & stone<!BLACK>();
        }
      }
    }
    isKo = maybeKo && size(capture) == 1;
  }
  int newKoPos = isKo ? firstOf(capture) : 0;
  return hash.update<BLACK>(pos, koPos, newKoPos, nPass, newNPass, capture); 
}

template<bool BLACK> void Node::playInt(int pos) {
  if (pos == PASS) {
    if (koPos) {
      assert(nPass == 0);
      koPos = 0;
    } else {
      assert(nPass < 2);
      ++nPass;
    }
    return;
  }
  
  assert(isEmpty(pos));
  assert(pos != koPos);
  nPass = 0;
    
  uint64_t group = shadow(pos);
  int newGid = -1;
  bool isSimple = true;
  bool maybeKo = true;
  uint64_t capture = 0;
  for (int p : NEIB(pos)) {
    if (isEmpty(p)) {
      maybeKo = false;
    } else {
      int gid = gids[p];
      if (is<BLACK>(p)) {
        maybeKo = false;
        group |= groups[gid];
        if (newGid == -1) {
          newGid = gid;
        } else if (newGid != gid) {
          isSimple = false;
          groups[gid] = 0;
        }
      } else if (is<!BLACK>(p)) {
        uint64_t g = groups[gid];
        if (libsOfGroup(g) == 1) {
          capture |= g & stone<!BLACK>();
          groups[gid] = 0;
        }
      }
    }
  }
  bool isKo = maybeKo && size(capture) == 1;
  koPos = isKo ? firstOf(capture) : 0;
  if (BLACK) {
    stoneWhite &= ~capture;
    SET(pos, stoneBlack);
  } else {
    stoneBlack &= ~capture;
    SET(pos, stoneWhite);
  }
  updateEmpty();
  if (newGid == -1) { newGid = this->newGid(); }
  groups[newGid] = group;
  if (isSimple) {
    gids[pos] = newGid;
  } else {
    updateGroupGids<BLACK>(group, newGid);
  }
  assert(libsOfGroupAtPos(pos) > 0);
  if (BLACK) {
    pointsBlack = bensonAlive<true>();
  } else {
    pointsWhite = bensonAlive<false>();
  }
}

template<bool BLACK> unsigned Node::neibGroups(int p) const {
  unsigned bits = 0;
  for (int pp : NEIB(p)) {
    if (is<BLACK>(pp)) { SET(gids[pp], bits); }
  }
  return bits;
}

struct Region {
  unsigned border;
  Vect<byte, 4> vital;
  uint64_t area;

  Region(): border(0), vital(), area(0) { }
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

static char *expand(char *line) {
  for (int i = SIZE_X - 1; i >= 0; --i) {
    line[2*i+1] = line[i];
    line[2*i] = ' ';
  }
  line[SIZE_X * 2] = 0;
  return line;
}

void printArea(uint64_t area) {
  char line[256];
  for (int y = 0; y < SIZE_Y; ++y) {
    for (int x = 0; x < SIZE_X; ++x) {
      int p = P(y, x);
      line[x] = IS(p, area) ? '+' : '.';
    }
    line[SIZE_X * 2] = 0;
    printf("\n%s", expand(line));
  }
  printf("\n");
}

std::pair<uint64_t, uint64_t> Node::enclosedRegions() const {
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
        } else if (is<true>(p)) {
          touchesBlack = true;
        } else if (is<false>(p)) {
          touchesWhite = true;
        }
        return false;
      });
    emptyNotSeen &= ~seen;
    assert(touchesBlack || touchesWhite);
    if (touchesBlack && !touchesWhite) {
      enclosedBlack |= area;
    } else if (touchesWhite && !touchesBlack) {
      enclosedWhite |= area;
    }
  }
  return std::make_pair(enclosedBlack, enclosedWhite);
}

template<bool BLACK> uint64_t Node::bensonAlive() const {
  Vect<Region, MAX_GROUPS> regions;
  uint64_t borderOrCol = BORDER | stone<BLACK>();
  uint64_t emptyNotSeen = empty;

  while (emptyNotSeen) {
    int p = firstOf(emptyNotSeen);
    unsigned vital = -1;
    unsigned border = 0;
    uint64_t area = 0;
    uint64_t seen = walk(p, [&area, &vital, &border, borderOrCol, this](int p) {
        if (IS(p, borderOrCol)) { return false; }
        unsigned gidBits = neibGroups<BLACK>(p);
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

  // printf("alive gids %x\n", aliveBits);  
  uint64_t points = 0;
  if (!aliveGids.isEmpty()) {
    for (Region &r : regions) {
      if (r.isCoveredBy(aliveBits) && !hasEyeSpace<!BLACK>(r.area)) {
        // printf("r area %lx\n", r.area);
        points |= r.area;
      }
    }
    uint64_t stonePoints = 0;
    for (int gid : aliveGids) {
      stonePoints |= groups[gid];
    }
    points |= (stonePoints & stone<BLACK>());
  }
  return points;
}

template<bool BLACK> bool Node::hasEyeSpace(uint64_t area) const {
  if (size(area) < 8) { return false; }
  int nEyes = 0;
  int firstEye = 0;
  for (int p : Bits(area & empty)) {
    if (!(shadow(p) & stone<!BLACK>())) {
      if (++nEyes >= 3) { return true; }
      if (!firstEye) {
        firstEye = p;
      } else if (!(shadow(p) & (1 << firstEye))) {
        return true;
      }
    }
  }
  return false;
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

template<typename T> bool Node::isSymmetry(T t) const {
  return stoneBlack == t(stoneBlack) && stoneWhite == t(stoneWhite);
}

uint64_t Node::maybeMoves() const {
  uint64_t area = empty;
  if (!koPos && size(empty) > N - 12) {
    if (isSymmetry(reflectX))    { area &= HALF_X; }
    if (isSymmetry(reflectY))    { area &= HALF_Y; }
    if (isSymmetry(reflectDiag)) { area &= HALF_DIAG; }
  }
  return area;
}

template<bool BLACK> void Node::genMoves(Vect<byte, N+1> &moves) const {
  assert(!isEnded());
  int tmp[N];
  int n = 0;
  uint64_t area = maybeMoves();
  if (koPos) { area &= ~(1ull << koPos); }
  for (int p : Bits(area)) {
    int v = valueOfMove<BLACK>(p);
    if (v >= 0) {
      tmp[n++] = ((1000 - v) << 8) | p;
    }
  }
  std::sort(tmp, tmp + n);
  moves.clear();
  if (nPass == 1) {
    assert(!koPos);
    moves.push(PASS);
  }
  for (int *pt = tmp, *end = tmp + n; pt < end; ++pt) {
    moves.push(*pt & 0xff);
  }
  if (nPass == 0) {
    moves.push(PASS);
  }
}

int Node::finalScore() const {
  assert(!(pointsBlack & pointsWhite));
  if (!stoneBlack && !stoneWhite) { return 0; }
  uint64_t enclosedBlack, enclosedWhite;
  std::tie(enclosedBlack, enclosedWhite) = enclosedRegions();
  uint64_t totalBlack = pointsBlack | ((stoneBlack | enclosedBlack) & ~pointsWhite);
  uint64_t totalWhite = pointsWhite | ((stoneWhite | enclosedWhite) & ~pointsBlack);
  assert((totalBlack & totalWhite) == 0);
  return size(totalBlack) - size(totalWhite);
}

template<bool MAX> Value Node::score(int beta) const {
  if (isEnded()) { return Value::makeExact(finalScore()); }
  if (!pointsBlack && !pointsWhite) { return Value::makeNone(); }
  
  int min = -N + 2 * size(pointsBlack);
  int max =  N - 2 * size(pointsWhite);
  assert(min <= max);
  /*
  if (nPass == 1) {
    int final = finalScore();
    assert(min <= final && final <= max);    
    if (MAX) {
      min = final;
    } else {
      max = final;
    }
  }
  */
  if (max < beta) { return Value::makeUpp(max); }
  return min == -N ? Value::makeNone() : Value::makeLow(min);
}

  /*
  } else {
    if (MAX) {
      return Value::makeLow(min);
    } else {
      return Value::makeUpp(max);
    }
    }*/

char Node::charForPos(int p) const {
  return is<true>(p) ? 'x' : is<false>(p) ? 'o' : isEmpty(p) ? '.' : isBorder(p) ? '-' : '?';
}

void Node::print(const char *s) const {
  if (s) { printf("%s\n", s); }
  char line1[256], line2[256], line3[256];
  for (int y = 0; y < SIZE_Y; ++y) {
    for (int x = 0; x < SIZE_X; ++x) {
      int p = P(y, x);
      line1[x] = charForPos(p);
      line2[x] = '0' + gids[p];
      bool isPointBlack = IS(p, pointsBlack);
      bool isPointWhite = IS(p, pointsWhite);
      assert(!(isPointBlack && isPointWhite));
      line3[x] = isPointBlack ? 'x' : isPointWhite ? 'o' : '.';
    }
    line1[SIZE_X*2] = 0;
    line2[SIZE_X*2] = 0;
    printf("\n%s    %s    %s", expand(line1), expand(line2), expand(line3));
  }
  /*
  printf("\n\nGroups:\n");
  for (int gid = 0; gid < MAX_GROUPS; ++gid) {
    if (groups[gid]) {
      int col = groupColor(gid);
      int size = (col == BLACK) ? sizeOfGid<BLACK>(gid) : sizeOfGid<WHITE>(gid);
      printf("%d size %d libs %d\n", gid, size, libsOfGid(gid));
    }
  }
  */
  if (koPos) { printf("ko: (%d, %d) ", Y(koPos), X(koPos)); }
  if (nPass) { printf("nPass %d ", nPass); }
  printf("\n\n");
}

#define TEMPLATES(BLACK) \
  template uint64_t Node::bensonAlive<BLACK>() const;   \
  template void Node::genMoves<BLACK>(Vect<byte, N+1> &) const;   \
  template int Node::valueOfMove<BLACK>(int) const;             \
  template Hash Node::hashOnPlay<BLACK>(const Hash &, int) const; \
  template Value Node::score<BLACK>(int) const;

TEMPLATES(true);
TEMPLATES(false);

#undef TEMPLATES

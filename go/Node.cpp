// (c) Copyright 2014 Mihai Preda. All rights reserved.

#include "Node.hpp"
#include "data.hpp"
#include "Hash.hpp"
#include "Value.hpp"
#include "go.hpp"

#include <initializer_list>
#include <algorithm>
#include <vector>
#include <utility>

#include <assert.h>
#include <stdio.h>

Node::Node() :
  hash(0),
  stoneBlack(0),
  stoneWhite(0),
  empty(INSIDE),
  pBlack(-1),
  pWhite(-1),
  koPos(0),
  nPass(0),
  swapped(false),
  gids{0},
{
}

/* The four points around <p>, not including <p>. */
static uint64_t shadowExcl(int p) {
  return ((uint64_t)(1 | (5 << (BIG_X-1)) | (1 << (BIG_X + BIG_X)))) << (p - BIG_X);
}

/* The shadow including <p>. */
static uint64_t shadowIncl(int p) {
  return shadowExcl(p) | (1 << p);
  // return ((uint64_t)(1 | (7 << (BIG_X-1)) | (1 << (BIG_X + BIG_X)))) << (p - BIG_X);
}

Node play(int p) const {
  return Node(*this).swapAndPlay(p);
}

void Node::swapAndPlay(Groups &groups, int pos) {
  swapped = !swapped;
  std::swap(stoneBlack, stoneWhite);
  pWhite = pBlack;
  pBlack = -1;
  
  assert(!(koPos && nPass));
  if (pos == PASS) {
    if (koPos) {
      koPos = 0;
    } else {
      assert(nPass < 2);
      ++nPass;
    }
  } else {  
    assert(isEmpty(pos) && pos != koPos);
    nPass = 0;
    
    uint64_t group = shadowIncl(pos);
    int newGid = -1;
    bool isSimple = true;
    bool maybeKo = true;
    uint64_t capture = 0;
    for (int p : NEIB(pos)) {
      if (isEmpty(p)) {
        maybeKo = false;
      } else {
        int gid = gids[p];
        if (isBlack(p)) {
          maybeKo = false;
          if (newGid != gid) {
            group |= groups[gid];
            if (newGid == -1) {
              newGid = gid;
            } else {
              isSimple = false;
              groups.release(gid);
            }
          }
        } else if (isWhite(p)) {
          uint64_t g = groups[gid];
          if (libsOfGroup(g) == 1) {
            capture |= g & stoneWhite;
            groups.release(gid);
          }
        }
      }
    }
    bool isKo = maybeKo && size(capture) == 1;
    koPos = isKo ? firstOf(capture) : 0;
    stoneWhite &= ~capture;
    SET(pos, stoneBlack);
    updateEmpty();
    if (newGid == -1) { newGid = this->newGid(); }
    groups[newGid] = group;
    if (isSimple) {
      gids[pos] = newGid;
    } else {
      setGroupGid(group, newGid);
    }
    assert(libsOfGroupAtPos(pos) > 0);
  }

  int codedKo = 0;
  if (koPos != 0) {
    ++codeKo;
    int start = P(0, 0);
    uint64_t shadowEx = shadowExcl(start);
    for (int p = start; p < koPos; ++p, shadowEx <<= 1) {
      if (IS(p, empty) && ((shadowEx & stoneBlack) == shadowEx)) {
        int nUnitCapture = 0;
        for (int pp : NEIB(p)) {
          int64_t group = groups[gids[pp]];
          if (sizeOfGroupBlack(group) == 1 && libsOfGroup(group) == 1) {
            ++nUnitCapture;
          }
        }
        if (nUnitCapture == 1) { // KO found.
          ++codeKo;
          if (codeKo > 3) { break; }
        }
      }
    }
    assert(codeKo <= 3);
  }
  hash = hashOf(stoneBlack, stoneWhite, codedKo, nPass != 0, swapped);
}

void Node::setGroupGid(uint64_t group, int gid) {
  for (int p : Bits(group)) { gids[p] = gid; }
}

int Node::newGid() {
  groups.push_back(0);
  return groups.size() - 1;
}

int Node::releaseGid(int gid) {
  assert(gid >= 0 && gid < groups.size());
  if (gid != groups.size() - 1) {
    setGroupGid(groups.back(), gid);
  }
  groups.pop_back();
}

#define NEIB(p) {p+1, p+DELTA, p-1, p-DELTA}

int Node::valueOfMoveBlack(int pos) const {
  assert(isEmpty(pos) && koPos != pos);
  bool isEye = true;
  int value = 0;
  uint64_t mergeGroup = shadowIncl(pos);
  uint64_t capture = 0;
  int blackGid = -1;
  bool singleBlackGroup = true;
  
  for (int p : NEIB(pos)) {
    if (isEmpty(p)) {
      isEye = false;
    } else if (isBorder(p)) {
      // value -= 3; // tend to avoid play on border.
    } else {
      int gid = gids[p];
      uint64_t group = groups[gid];
      if (isBlack(p)) {
        singleBlackGroup &= (gid == (blackGid == -1 ? (blackGid = gid) : blackGid));        
        mergeGroup |= group;
      } else {
        assert(isWhite(p));
        isEye = false;
        int libs = libsOfGroup(group);        
        if (libs == 1) {
          capture |= group;
        } else if (libs == 2) {
          value += 10;
        }
      }
    }
  }
  // Never play inside a simple true eye. Play inside a fake eye with low priority.
  if (isEye) { return singleBlackGroup ? -1 : 0; }
  capture &= stoneWhite;
  uint64_t newEmpty = (empty & ~(1ull << pos)) | capture;
  int mergeLibs = size(mergeGroup & newEmpty);
  if (mergeLibs == 0) { return -1; } // suicide
  return value + (size(capture) + mergeLibs - 1) * 10;
}

template<bool BLACK> unsigned Node::neibGroupsBlack(int p) const {
#define FOO(p) (isBlack(p) ? (1 << gids[p]) : 0)
  return FOO(p + 1) | FOO(p + DELTA) | FOO(p - 1) | FOO(p - DELTA);
#undef FOO
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

template<bool BLACK> uint64_t Node::bensonAliveBlack() const {
  Vect<Region, MAX_GROUPS> regions;
  uint64_t borderOrCol = BORDER | stoneBlack;
  uint64_t emptyNotSeen = empty;

  while (emptyNotSeen) {
    int p = firstOf(emptyNotSeen);
    unsigned vital = -1;
    unsigned border = 0;
    uint64_t area = 0;
    uint64_t seen = walk(p, [&area, &vital, &border, borderOrCol, this](int p) {
        if (IS(p, borderOrCol)) { return false; }
        unsigned gidBits = neibGroupsBlack(p);
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
      if (r.isCoveredBy(aliveBits) && !hasEyeSpaceWhite(r.area)) {
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

bool Node::hasEyeSpaceWhite(uint64_t area) const {
  if (size(area) < 8) { return false; }
  int nEyes = 0;
  int firstEye = 0;
  for (int p : Bits(area & empty)) {
    if (!(shadowExcl(p) & stoneBlack())) {
      if (++nEyes >= 3) { return true; }
      if (!firstEye) {
        firstEye = p;
      } else if (!IS(firstEye, shadowExcl(p)) {
        return true;
      }
    }
  }
  return false;
}

void Node::genMovesBlack(Vect<byte, N+1> &moves) const {
  assert(!isEnded());
  int tmp[N];
  int n = 0;
  uint64_t exclude = pointsBlack() | pointsWhite() | (1ull << koPos);
  uint64_t area = empty & ~exclude;
  for (int p : Bits(area)) {
    int v = valueOfMoveBlack(p);
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

Value Node::score(int beta) const {
  if (isEnded()) { return Value::makeExact(finalScore()); }
  // if (!pointsBlack && !pointsWhite) { return Value::makeNone(); }
  
  int min = -N + 2 * size(pointsBlack);
  int max =  N - 2 * size(pointsWhite);
  assert(min <= max);
  /*
  if (nPass == 1) {
    int final = finalScore();
    assert(min <= final && final <= max);    
    min = final;
  }
  */
  return Value(min, max);
}

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

/*
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
*/


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
  nPass(0),
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

template<int C> int Node::valueOfMove(int pos) const {
  assert(isEmpty(pos) && koPos != pos);
  bool isSuicide = true;
  bool isSelfEye = true;
  int value = 0;
  int prevGid = -1;
  for (int p : NEIB(pos)) {
    if (isEmpty(p)) {
      isSuicide = false;
      isSelfEye = false;
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
          isSelfEye = false;
          value += 2;
        } else if (libs == 2) {
          ++value;
        }
        if (gid != prevGid) {
          prevGid = gid;
        } else {
          --value;
        }
      } else if (is<1-C>(p)) {
        isSelfEye = false;
        if (libs == 1) {
          isSuicide = false;
          value += sizeOfGroup<1-C>(group) + 1;
        } else if (libs == 2) {
          ++value;
        }
      }
    }
  }
  assert(value >= -2);
  return (isSuicide || isSelfEye) ? -1 : (value + 2);
}

template<int C> uint128_t Node::hashOnPlay(int pos) const {
  uint128_t newHash = hash ^ hashSide();
  if (pos == PASS) {
    if (koPos) {
      assert(nPass == 0);
      newHash ^= hashPass(1);
    } else {
      assert(nPass < 3);
      if (nPass) { newHash ^= hashPass(nPass); }
      newHash ^= hashPass(nPass ? (nPass + 1) : 2);
    }
  }
  
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

  assert(!isKo || koPos != pos);
  if (koPos) { newHash ^= hashKo(koPos); }
  if (isKo) { newHash ^= hashKo(pos); }
  if (capture) { for (int p : Bits(capture & stone[1-C])) { newHash ^= hashPos<1-C>(p); } }
  return newHash;
}

void Node::setKoPos(int p) {
  if (koPos)       { hash ^= hashKo(koPos); }
  if ((koPos = p)) { hash ^= hashKo(koPos); }
}

void Node::setNPass(int n) {
  if (nPass)       { hash ^= hashPass(nPass); }
  if ((nPass = n)) { hash ^= hashPass(nPass); }
}

template<int C> void Node::playInt(int pos) {
  static_assert(isBlackOrWhite(C), "color");
  hash ^= hashSide();
  
  if (pos == PASS) {
    if (koPos) {
      assert(nPass == 0);
      setKoPos(0);
      setNPass(1);
    } else {
      assert(nPass < 3);
      setNPass((nPass == 0) ? 2 : (nPass + 1));
    }
    return;
  }
  
  assert(isEmpty(pos));
  if (nPass) {
    assert(nPass < 3);
    setNPass(0);
  }
  
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
  if (koPos || isKo) { setKoPos(isKo ? pos : 0); }
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

template<int C> unsigned Node::neibGroups(int p) const {
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

void Node::enclosedRegions(uint64_t *outEnclosed) const {
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

template<int C> uint64_t Node::bensonAlive() const {
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

template<typename T> bool Node::isSymmetry(T t) const {
  return stone[BLACK] == t(stone[BLACK]) && stone[WHITE] == t(stone[WHITE]);
}

uint64_t Node::maybeMoves() const {
  uint64_t area = INSIDE;
  if (!koPos && size(empty) > N - 12) {
    if (isSymmetry(reflectX))    { area &= HALF_X; }
    if (isSymmetry(reflectY))    { area &= HALF_Y; }
    if (isSymmetry(reflectDiag)) { area &= HALF_DIAG; }
  }
  return area;
}

template<int C> void Node::genMoves(Vect<byte, N> &moves) const {
  assert(nPass < 3);
  int tmp[N];
  int n = 0;
  uint64_t area = maybeMoves();
  if (koPos) { area &= ~(1 << koPos); }
  for (int p : Bits(area)) {
    int v = valueOfMove<C>(p);
    if (v >= 0) {
      tmp[n++] = ((1000 - v) << 8) | p;
    }
  }
  std::sort(tmp, tmp + n);
  moves.clear();
  if (nPass == 2) {
    assert(!koPos);
    moves.push(PASS);
  }
  for (int *pt = tmp, *end = tmp + n; pt < end; ++pt) {
    moves.push(*pt & 0xff);
  }
  if (nPass < 2) {
    moves.push(PASS);
  }
}

template<int C> int Node::finalScore() const {
  uint64_t enclosed[2] = {0};
  enclosedRegions(enclosed);
  uint64_t total[2];
  for (int i : {BLACK, WHITE}) {
    total[i] = points[i] | stone[i] | enclosed[i];
  }
  assert((total[BLACK] & total[WHITE]) == 0);
  return size(total[C]) - size(total[1-C]);
}

template<int C> ScoreBounds Node::score() const {
  if (nPass < 3) {
    return {(signed char) (-N + 2 * size(points[C])), (signed char) (N - 2 * size(points[1-C]))};
  } else {
    signed char score = finalScore<C>();
    return {score, score};
  }
}



static char *expand(char *line) {
  for (int i = SIZE_X - 1; i >= 0; --i) {
    line[2*i+1] = line[i];
    line[2*i] = ' ';
  }
  line[SIZE_X * 2] = 0;
  return line;
}

char Node::charForPos(int p) const {
  return is<BLACK>(p) ? 'x' : is<WHITE>(p) ? 'o' : isEmpty(p) ? '.' : isBorder(p) ? '-' : '?';
}

int Node::groupColor(int gid) const {
  for (int p = 0; p < BIG_N; ++p) {
    if (gids[p] == gid && (is<BLACK>(p) || is<WHITE>(p))) {
      return is<BLACK>(p) ? BLACK : WHITE;
    }
  }
  printf("groupColor gid %d %lx %d\n", gid, groups[gid], gids[P(0, 0)]);
  assert(false);
}

void Node::print() const {
  char line1[256], line2[256], line3[256];
  for (int y = 0; y < SIZE_Y; ++y) {
    for (int x = 0; x < SIZE_X; ++x) {
      int p = P(y, x);
      line1[x] = charForPos(p);
      line2[x] = '0' + gids[p];
      bool isPointBlack = IS(p, points[BLACK]);
      bool isPointWhite = IS(p, points[WHITE]);
      assert(!(isPointBlack && isPointWhite));
      line3[x] = isPointBlack ? 'x' : isPointWhite ? 'o' : '.';
    }
    line1[SIZE_X*2] = 0;
    line2[SIZE_X*2] = 0;
    printf("\n%s    %s    %s", expand(line1), expand(line2), expand(line3));
  }
  printf("\n\nGroups:\n");
  for (int gid = 0; gid < MAX_GROUPS; ++gid) {
    if (groups[gid]) {
      int col = groupColor(gid);
      int size = (col == BLACK) ? sizeOfGid<BLACK>(gid) : sizeOfGid<WHITE>(gid);
      printf("%d size %d libs %d\n", gid, size, libsOfGid(gid));
    }
  }
  if (koPos) {
    printf("ko: (%d, %d)\n", Y(koPos), X(koPos));
  }
  printf("\n\n");
}


#define TEMPLATES(C) \
  template void Node::playInt<C>(int);          \
  template uint64_t Node::bensonAlive<C>() const;   \
  template ScoreBounds Node::score<C>() const;      \
  template int Node::finalScore<C>() const;             \
  template void Node::genMoves<C>(Vect<byte, N> &) const;   \
  template int Node::valueOfMove<C>(int) const;             \
  template uint128_t Node::hashOnPlay<C>(int) const;

TEMPLATES(BLACK)
TEMPLATES(WHITE)

#undef TEMPLATES

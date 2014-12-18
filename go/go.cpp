#include "data.h"
#include "go.h"
#include <assert.h>
#include <stdio.h>

const uint64_t zob[128] = {
0x54d918dd5d2f0090ull, 0xb75f37ff7f4ad6c2ull, 0x63d1c96eb2586f32ull, 0x03c5f87034eaeb43ull, 
0xe0bbd16d8c0920ceull, 0x8cfa0698245226eaull, 0x75d16a1fe38cc36dull, 0xd10f09a578f5a73cull, 
0x45cb961a9a71a545ull, 0x23c7d169df814c15ull, 0x84c52c5548c7ae71ull, 0x9ced066f2fb08692ull, 
0x604a9747825958e9ull, 0x3b2fe622407e315bull, 0xcad7986f4654af33ull, 0xe460e73fe1e27cf0ull, 
0xb9344703756cfaefull, 0xaa40770ab31c830aull, 0xa880f6e28e65d039ull, 0xaec317811926c7e9ull, 
0xf150ba79062bea34ull, 0xd48e58f3116a4268ull, 0x9ea815f597e35da4ull, 0x7039b9e61e26c84aull, 
0x3eefdaad53adf2f8ull, 0x5f3d793b1deb4166ull, 0x14e47d06be950293ull, 0x62b8734d78e83bebull, 
0xf0b438d454d372beull, 0xaf17f718dd076d37ull, 0x75bd02aed736c479ull, 0x550638db61bc120eull, 
0xb28896c4e96740c1ull, 0xcb740e9ca5306883ull, 0x549e1d1f26957a84ull, 0xe405c3df0b54c052ull, 
0xcc46a7bfe0e355a9ull, 0xde33dfb0d28f0158ull, 0x3ed9f4017c627f3dull, 0xe124c9d8affbd805ull, 
0x8847f58df69647bbull, 0xd1b46f2964cde56cull, 0xb8f386038142df78ull, 0xf32cd5a0d9e0fdc6ull, 
0x5bd9db17065b2e4dull, 0x1048c66644311b84ull, 0x24b12b0fcb931d1full, 0x6947115c0a7067aeull, 
0xeace689ceeeb9bb2ull, 0xbf69352ba805e3daull, 0x8f9ffd46dcf0234dull, 0xe7241b68bf610361ull, 
0xd400ab94780f5c5aull, 0x015523d135c55c98ull, 0x5004e65eb2b37ccdull, 0xf1885c4eb8fff31cull, 
0x11b19b9e732adb95ull, 0x4369f17b9463db52ull, 0x967dd0f4f20c30d0ull, 0x79d46e3e8eb0835cull, 
0xc11191839fa454f8ull, 0xd0c37879f8721cc1ull, 0xb7c25bf2b1ad6ec2ull, 0xe0ca3dbcbccc917cull,
0x6a43c083b7a7d552ull, 0x87caad3d937b9b75ull, 0x0462397716b2cd36ull, 0x1341ef131df700ceull, 
0x6765bf7621089a4full, 0x58af8231a2714e0cull, 0x221438f13d217b0bull, 0x2b4e0a20d40d1778ull, 
0x4b9d7ab7790a64ecull, 0x3423f44cbbf22283ull, 0x0850597d42877e2bull, 0x07ac1679f3a8c52cull, 
0x18bcc6a2b6c5762bull, 0xbead75580d155a6bull, 0x434e448b1246ea0dull, 0x18f81a794b6d2418ull, 
0xa6c9c3bb41a0255eull, 0x7ef63ef3e815f981ull, 0xf30c3d5b7190a2efull, 0x4af673f860518ae8ull, 
0x4c4a868c1384bc6dull, 0xcb1947fafc459543ull, 0x34b2a31afd519b48ull, 0x1fc1cb75054c38d2ull, 
0x0224791aa952835full, 0x039acb66c01db264ull, 0x35b9303f964c64c7ull, 0x8d912909a7fcb42eull, 
0xcad5ffcf54b94855ull, 0x14b01f3cb0f6c850ull, 0x1fdc2ea98adc18dbull, 0xae006dd1ee658961ull, 
0x1ead788b67484b39ull, 0xca2dbf66080f0e80ull, 0x15b443db11f1b388ull, 0x90f7bb2f05cb3689ull, 
0x25fb04f12f5b4101ull, 0x93644e43fae4eb94ull, 0x55dd9ec6fe076abcull, 0x1bdee2d00abc4565ull, 
0x7e48a2bd4a3e5adaull, 0xb6842e93dd8595e6ull, 0x65b90fa62a9673e6ull, 0x52fc814f519c3e1bull, 
0x2e25123c733da701ull, 0x289a056664bf598full, 0xa59c30f1eb690b07ull, 0x447d59acd69cf651ull, 
0x00c0dd78c3714e32ull, 0x13417089a59ede0eull, 0x1fdb3e4a129b6368ull, 0x76e99a0cad62c9f1ull, 
0x139f5b641a8b463dull, 0x292b118ed8ce881aull, 0xa6de00a17f781120ull, 0x026f8bc108d64edeull, 
0x6b462d684bc4d9dbull, 0xf3bcfee3f4692ec6ull, 0x57df6d044ae10effull, 0x3aef7658da36d76aull, 
0xd1f0311d2b5e403cull, 0x510b773fe3223973ull, 0xb22aba50f53fa7bcull, 0x7df1caec709f35b8ull,
};

uint64_t hashPos(int p, int color) { return zob[(p << 1) + color]; }
uint64_t hashKo(int p) { return hashPos(p, BLACK) ^ hashPos(p, WHITE); }
uint64_t hashSideToPlay(int color) { return hashPos(0, color); }

int pos(int y, int x) { return (y + 1) * BIG_X + x + 1; }
bool isBlackOrWhite(int color) { return color == BLACK || color == WHITE; }
bool isValid(int y, int x) { return y >= 0 && y < SIZE_Y && x >= 0 && x < SIZE_X; }

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
public:
  Cell cells[BIG_N];
  Group groups[MAX_GROUPS];
  uint64_t hash;

  Board();
  
  int color(int p) const { return cells[p].color; }
  Group *group(int p) { return groups + cells[p].group; }
  Group *newGroup();
  int libs(int p) { return group(p)->libs; }
  int groupColor(const Group &g) { return color(g.pos); }
  
  bool move(int p, int color);
  
  void updateGroup(int p, int gid);
  void updateGroupLibs(int p);
  void removeGroup(int p);
  
  unsigned neibGroupsOfColor(int p, int col);
  void bensonAlive(int col, Bitset &points, unsigned *outAliveBits);
  
  void print(const Bitset &, const Bitset &);
};

struct State {
  Board board;
  int ko;

  State() : board(), ko(0) { }
};

Board::Board() {
  for (int y = 0; y < SIZE_Y + 1; ++y) {
    cells[y * BIG_X].color = BROWN;
    cells[y * BIG_X + SIZE_X + 1].color = BROWN;
  }
  for (int x = 0; x < BIG_X; ++x) {
    cells[x].color = BROWN;
    cells[(SIZE_Y + 1) * BIG_X + x].color = BROWN;
  }
  hash = hashSideToPlay(BLACK);
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

bool Board::move(int pos, int col) {
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
  Group *g = 0;
  for (int p : neighb) {
    if (color(p) == col) {
      if (!g) {
        g = group(p);
      } else if (group(p) != g) {
        group(p)->size = 0;
      }
    } else if (color(p) == otherCol) {
      --group(p)->libs;
    }
  }
  if (!g) { g = newGroup(); }
  int gid = g - groups;
  cells[pos] = Cell(col, gid);
  updateGroup(pos, gid);
  return true;
}

char charForColor(int color) { return color == BLACK ? 'x' : color == WHITE ? 'o' : '.'; }

char *expand(char *line) {
  for (int i = SIZE_X - 1; i >= 0; --i) {
    line[2*i+1] = line[i];
    line[2*i] = ' ';
  }
  line[SIZE_X * 2] = 0;
  return line;
}

void Board::print(const Bitset &pointsBlack, const Bitset &pointsWhite) {
  char line1[256], line2[256], line3[256];
  for (int y = 0; y < SIZE_Y; ++y) {
    for (int x = 0; x < SIZE_X; ++x) {
      int p = pos(y, x);
      Cell c = cells[p];
      line1[x] = charForColor(c.color);
      line2[x] = '0' + c.group;
      bool isPointBlack = pointsBlack.test(p);
      bool isPointWhite = pointsWhite.test(p);
      assert(!(isPointBlack && isPointWhite));
      line3[x] = charForColor(isPointBlack ? BLACK : isPointWhite ? WHITE : EMPTY);
    }
    line1[SIZE_X*2] = 0;
    line2[SIZE_X*2] = 0;
    printf("\n%s    %s    %s", expand(line1), expand(line2), expand(line3));
  }
  printf("\n\nGroups:\n");
  for (int gid = 0; gid < MAX_GROUPS; ++gid) {
    Group g = groups[gid];
    if (g.size > 0) {
      printf("%d size %d libs %d pos %d\n", gid, g.size, g.libs, g.pos);
    }
  }
  printf("\n\n");
}

int main() {
  Board b;
  printf("%llx\n", b.hash);
  Bitset pointsMe, pointsOth;
  b.print(pointsMe, pointsOth);
  while (true) {
    char buf[16] = {0};
    int y = -1;
    int x = -1;
    printf("> ");
    if (scanf("%1s %1d %1d", buf, &y, &x) != 3) { continue; }
    char c = buf[0];
    int col = c == 'b' ? BLACK : c == 'w' ? WHITE : EMPTY;
    if (isBlackOrWhite(col) && isValid(y, x) && b.color(pos(y, x)) == EMPTY) {
      if (!b.move(pos(y, x), col)) {
        printf("suicide\n");
      }

      unsigned aliveGroupBitsMe, aliveGroupBitsOth;
      b.bensonAlive(col, pointsMe, &aliveGroupBitsMe);
      b.bensonAlive((1-col), pointsOth, &aliveGroupBitsOth);
      if (col == BLACK) {
        b.print(pointsMe, pointsOth);
      } else {
        b.print(pointsOth, pointsMe);
      }
    }
  }
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
  
  void print() {
    printf("region border %x area %d vital ", border, area.size());
    for (int gid : vital) {
      printf("%d ", gid);
    }
    printf("\n");
  }

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

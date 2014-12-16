#include "data.h"
#include "go.h"
#include <assert.h>
#include <stdio.h>

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

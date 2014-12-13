#include "go.h"

#include <bitset>
#include <cassert>
#include <cstdio>
#include <string>

using namespace std;

const int DELTAS[] = {1, -1, DELTA_UP, DELTA_DOWN};

int pos(int y, int x) { return (y + 1) * BIG_X + x + 1; }

#if SIZE_X == 6 and SIZE_Y == 6
#define REP(a) pos(a, 0), pos(a, 1), pos(a, 2), pos(a, 3), pos(a, 4), pos(a, 5)
static const int idx[N] = { REP(0), REP(1), REP(2), REP(3), REP(4), REP(5) };
#undef REP
#endif

const auto Black = [](int color) { return color == BLACK; };
const auto Empty = [](int color) { return color == EMPTY; };

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
    pos((byte) pos) { }
  
  byte size;
  byte libs;
  byte pos;
};

template<typename T> class Region;
template<typename R, typename T> class Region2;

template<typename R>
void print(R r) {
  for (int p : r) {
    printf("%d ", p);
  }
  printf("\n");
}

template<typename R> bool isEmpty(R r) { return !(r.begin() != r.end()); }

template<typename R, typename T> Region2<R, T> border(R reg, T accept) {
  return Region2<R, T>(reg.board, reg, accept);
}

template<typename R> bool isDead(R region) {
  return !isEmpty(region) && isEmpty(border(region, Empty));
}

template<typename T>
int set(Region<T> r, Cell cell) {
  Cell *cells = r.board->cells;
  int n = 0;
  for (int p : r) {
    cells[p] = cell;
    ++n;
  }
  return n;
}

class Board {
public:
  Cell cells[BIG_N];
  Group groups[MAX_GROUPS];

  Board();
  int color(int p) const { return cells[p].color; }
  bool move(int p, int color);

  template<typename T> auto region(int start, T accept);
  auto regionOfColor(int start, int color);

  int nearGroupsOfColor(int p, int color, int *outGroups);
  int nearLibs(int p);
  int newGid();
  int remove(int p, int color);
  int libs(int p);
  int libsAndSetGroup(int p, int group);
  
  void print();
};

template<typename T, int openSize>
class BaseIt {
protected:
  Cell * const cells;
  const T accept;
  int size;
  bitset<BIG_N> seen;
  byte open[openSize];

  void add(int p) {
    int color = cells[p].color;
    if (accept(color) && color != BROWN && !seen[p]) {
      seen[p] = true;
      open[size++] = p;
    }
  }

  void addNeighbours(int p) {
    add(p + 1);
    add(p - 1);
    add(p + DELTA_DOWN);
    add(p + DELTA_UP);
  }

  bool atEnd() const { return size <= 0; }

  BaseIt(Cell *cells, T accept) : cells(cells), accept(accept), size(0) { } 

public:
  int operator*() {
    assert(!atEnd());
    return open[size - 1];
  }

  bool operator!=(const BaseIt &other) { return atEnd() != other.atEnd(); }
};

template<typename R, typename T>
class Region2 {
private:  
  class Iterator : public BaseIt<T, 4> {
  private:
    typedef typename R::Iterator SubIt;
    SubIt it, itEnd;

    void lookAhead() {
      while (this->atEnd() && it != itEnd) {
        this->addNeighbours(*it);
        ++it;
      }
    }
        
  public:
    Iterator(Cell *cells, T accept, SubIt it, SubIt itEnd) :
      BaseIt<T, 4>(cells, accept), it(it), itEnd(itEnd) {
      lookAhead();
    }

    void operator++() {
      assert(!this->atEnd());
      --this->size;
      if (this->atEnd()) { lookAhead(); }
    }
  };

  Board *board;
  R subreg;
  const T accept;
      
public:
  Region2(Board *b, R subreg, T accept) : board(b), subreg(subreg), accept(accept) { }
  Iterator begin() { return Iterator(board->cells, accept, subreg.begin(), subreg.end()); }
  Iterator end()   { return Iterator(board->cells, accept, subreg.end(), subreg.end()); }
};


template<typename T>
class Region {
public:
  Board *board;
  const int start;
  const T accept;

public:
  class Iterator: public BaseIt<T, N> {
  public:
    Iterator(Cell *cells, T accept, int start) : BaseIt<T, N>(cells, accept) { this->add(start); }
    Iterator(T accept) : BaseIt<T, N>(0, accept) { }

    void operator++() {
      int p = this->operator*();
      --this->size;
      this->addNeighbours(p);
    }
  };

  Region(Board *b, int start, T accept) : board(b), start(start), accept(accept) { }
  Iterator begin() { return Iterator(board->cells, accept, start); }
  Iterator end()   { return Iterator(accept); }
};

template<typename T>
auto Board::region(int start, T accept) {
  return Region<T>(this, start, accept);
}

auto Board::regionOfColor(int start, int color) {
  return region(start, [color](int c) { return c == color; });
}

Board::Board() {
  for (int y = 0; y < SIZE_Y + 1; ++y) {
    cells[y * BIG_X].color = BROWN;
    cells[y * BIG_X + SIZE_X + 1].color = BROWN;
  }
  for (int x = 0; x < BIG_X; ++x) {
    cells[x].color = BROWN;
    cells[(SIZE_Y + 1) * BIG_X + x].color = BROWN;
  }
  // groups[0] = Group((BIG_X + SIZE_Y) * 2, 2, 0);
}

template<typename T, int N>
class Vect {
  T v[N];
  int size = 0;

public:
  void push(T t) { v[size++] = t; }
  T pop() { return v[--size]; }
  bool isEmpty() { return size <= 0; }
};

class Bitset {
  unsigned long long bits = 0;
public:
  bool testAndSet(int p) {
    unsigned long long mask = (1ull << p);
    bool bit = bits & mask;
    if (!bit) { bits |= mask; }
    return bit;
  }
};

#define STEP(inip) { int p = inip; if (!seen.testAndSet(p)) { if (cells[p].color == color) { open.push(p); PROCESS; } else { ELSE; } } }

#define WALK(p) open.push(p); PROCESS; do { int p = open.pop(); STEP(p+1); STEP(p-1); STEP(p+DELTA_UP); STEP(p-DELTA_UP); } while (!open.isEmpty());

int Board::libsAndSetGroup(int p, int group) {
#define PROCESS cells[p] = cell
#define ELSE if (cells[p].color == EMPTY) { ++n; }
  Bitset seen;
  Vect<byte, N> open;
  int color = cells[p].color;
  assert(color == WHITE || color == BLACK);
  int n = 0;
  Cell cell(color, group);
  WALK(p);
  return n;
#undef PROCESS
#undef ELSE
}

int Board::libs(int p, int group) {
#define PROCESS
#define ELSE if (cells[p].color == EMPTY) { ++n; }
  Bitset seen;
  Vect<byte, N> open;
  int color = cells[p].color;
  assert(color == WHITE || color == BLACK);
  int n = 0;
  Cell cell(color, group);
  WALK(p);
  return n;
#undef PROCESS
#undef ELSE
}

int Board::nearGroupsOfColor(int p, int color, int *out) {
  int n = 0;
  out[0] = out[1] = out[2] = out[3] = 0;
  for (int delta : DELTAS) {
    int pp = p + delta;
    Cell c = cells[pp];
    if (c.color == color) {
      int g = c.group;
      if (out[0] == g || out[1] == g || out[2] == g) { continue; }
      out[n++] = g;
    }
  }
  return n;
}

int Board::nearLibs(int p) {
  int n = 0;
  for (int delta : DELTAS) { if (cells[p + delta].color == EMPTY) { ++n; } }
  return n;
}

int Board::newGid() {
  for (int i = 1; i < MAX_GROUPS; ++i) {
    if (groups[i].size == 0) {
      return i;
    }
  }
  assert(false && "max groups exceeded");
}

int Board::remove(int p, int color) {
  return set(regionOfColor(p, color), Cell(EMPTY, 0));
}

bool Board::move(int p, int color) {
  Cell &cell = cells[p];
  assert(cell.color == EMPTY);
  assert(color == WHITE || color == BLACK);
  int otherCol = 1 - color;
  bool hasRemoved = false;
  for (int delta : DELTAS) {
    int pp = p + delta;
    if (cells[pp].color == otherCol) {
      int gid = cells[pp].group;
      if (groups[gid].libs == 1) {
        remove(pp, otherCol);
        groups[gid].size = 0;
        hasRemoved = true;
      }
    }
  }
  if (hasRemoved) {
    for (int gid = 0; gid < MAX_GROUPS; ++gid) {
      if (groups[gid].size > 0) {
        
      }
      Group &g = groups[gid];
      if (g.size > 0 && cells[g.pos].color == color) {
        g.libs = nLibs(g.pos);
      }
    }

  }
  
  for (int gid : gids) {
    
  }
  
  
  int gidsOther[4];
  int nOther = nearGroupsOfColor(p, otherCol, gidsOther);
  bool anyRemoved = false;
  for (int i = 0; i < nOther; ++i) {
    int gid = gidsOther[i];
    Group &group = groups[gid];
    assert(group.size > 0 && group.libs > 0);
    if (--group.libs == 0) {
      int size = remove(group.pos, otherCol);
      assert(size == group.size);
      group.size = 0;
      anyRemoved = true;      
    }
  }
  
  if (anyRemoved) {
  }

  int newLibs = nearLibs(p);

  int gidsSame[4];
  int nSame = nearGroupsOfColor(p, color, gidsSame);
  printf("other %d same %d libs %d\n", nOther, nSame, newLibs);

  int sumLibs = 0;
  int sumSize = 0;
  for (int i = 0; i < nSame; ++i) {
    int gid = gidsSame[i];
    sumLibs += groups[gid].libs;
    sumSize += groups[gid].size;
  }
  int libs = sumLibs + newLibs - nSame;
  assert(libs >= 0);
  if (libs == 0) {
    for (int i = 0; i < nOther; ++i) {
      ++groups[gidsOther[i]].libs;
    }
    return false;
  }
  int gid = nSame == 0 ? newGid() : gidsSame[0];
  groups[gid] = Group(sumSize + 1, libs, p);
  for (int i = 1; i < nSame; ++i) {
    Group &g = groups[gidsSame[i]];
    for (int pp : regionOfColor(g.pos, color)) {
      cells[pp].group = gid;
    }
    g.size = 0;
  }
  cell = Cell(color, gid);
  return true;
}

char charForColor(int color) {
  return color == BLACK ? 'x' : color == WHITE ? 'o' : '.';
}

void Board::print() {
  std::string line1, line2;
  for (int y = 0; y < SIZE_Y; ++y) {
    line1.clear();
    line2.clear();
    for (int x = 0; x < SIZE_X; ++x) {
      Cell c = cells[pos(y, x)];
      line1 += ' ';
      line1 += charForColor(c.color);
      line2 += ' ';
      line2 += '0' + c.group;
    }
    printf("\n%s    %s", line1.c_str(), line2.c_str());
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

bool valid(int y, int x) { return y >= 0 && y < SIZE_Y && x >= 0 && x < SIZE_X; }

int main() {
  Board b;
  b.print();
  while (true) {
    char buf[16] = {0};
    int y = -1;
    int x = -1;
    printf("> ");
    scanf("%1s %1d %1d", buf, &y, &x);
    char c = buf[0];
    int color = c == 'b' ? BLACK : c == 'w' ? WHITE : EMPTY;
    if ((color == BLACK || color == WHITE) && valid(y, x) && b.color(pos(y, x)) == EMPTY) {
      b.move(pos(y, x), color);
      b.print();
    }
  }
  
  /*  
  b.move(pos(1, 1), BLACK);
  b.move(pos(0, 1), BLACK);
  b.print();

  auto black = b.region(pos(1, 1), Black);
  print(black);
  auto libs = border(black, Empty);
  print(libs);
  printf("Dead %d\n", isDead(black));
  */
}

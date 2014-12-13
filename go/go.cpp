#include "go.h"

#include <bitset>
#include <string>
#include <cassert>
#include <cstdio>

using namespace std;

const int DELTAS[] = {1, -1, -BIG_X, BIG_X};

int pos(int y, int x) { return (y + 1) * BIG_X + x + 1; }

#if SIZE_X == 6 and SIZE_Y == 6
#define REP(a) pos(a, 0), pos(a, 1), pos(a, 2), pos(a, 3), pos(a, 4), pos(a, 5)
static const int idx[N] = { REP(0), REP(1), REP(2), REP(3), REP(4), REP(5) };
#undef REP
#endif

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
  Group *group(int p) { return groups + cells[p].group; }
  Group *newGroup();
  int libs(int p) { return group(p)->libs; }
  int groupColor(const Group &g) { return color(g.pos); }
  
  bool move(int p, int color);
  int remove(int p, int color);
  
  void updateGroup(int p, int gid);
  void updateGroupLibs(int p);
  
  template<typename T> auto region(int start, T accept);
  auto regionOfColor(int start, int color);

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
    add(p + BIG_X);
    add(p - BIG_X);
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

template<typename T>
void walk(int p, T t) {
#define STEP(p) if (!seen.testAndSet(p) && t(p)) { open.push(p); }
  Bitset seen;
  Vect<byte, N> open;
  // if (t(p)) { open.push(p); }
  STEP(p);
  while (!open.isEmpty()) {
    const int p = open.pop();
    STEP(p + 1);
    STEP(p - 1);
    STEP(p + BIG_X);
    STEP(p - BIG_X);
  }
#undef STEP
}

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
  assert(col == BLACK || col == WHITE);  
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

Group *Board::newGroup() {
  for (Group *g = groups, *end = groups + MAX_GROUPS; g < end; ++g) {
    if (g->size == 0) { return g; }
  }
  assert(false && "max groups exceeded");
}

int Board::remove(int p, int color) {
  return set(regionOfColor(p, color), Cell(EMPTY, 0));
}

bool Board::move(int pos, int col) {
  assert(color(pos) == EMPTY);
  assert(col == WHITE || col == BLACK);
  int otherCol = 1 - col;
  int neighb[] = {pos+1, pos-1, pos+BIG_X, pos-BIG_X};
  bool captured = false;
  for (int p : neighb) {
    if (color(p) == otherCol && libs(p) == 1) {
      group(p)->size = 0;
      remove(p, otherCol);
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
      if (!b.move(pos(y, x), color)) {
        printf("suicide\n");
      }
      b.print();
    }
  }
}

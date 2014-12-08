#include <stdio.h>
#include <bitset>
#include <cassert>

typedef unsigned char byte;
using namespace std;

enum {
  SIZE_Y = 6,
  SIZE_X = 6,

  BIG_X = SIZE_X + 2,
  BIG_Y = SIZE_Y + 2,
  
  N = SIZE_X * SIZE_Y,
  BIG_N = BIG_X * BIG_Y,

  DELTA_UP = -BIG_X,
  DELTA_DOWN = BIG_X
};

enum Color {
  BLACK = 0,
  WHITE = 1,
  EMPTY = 2,
  BROWN = 3
};

static int pos(int y, int x) { return (y + 1) * BIG_X + x + 1; }

#if SIZE_X == 6 and SIZE_Y == 6
#define REP(a) pos(a, 0), pos(a, 1), pos(a, 2), pos(a, 3), pos(a, 4), pos(a, 5)
static const int idx[N] = { REP(0), REP(1), REP(2), REP(3), REP(4), REP(5) };
#undef REP
#endif

struct Block {  
};

struct Cell {
public:
  Cell():
    color(EMPTY),
    group(0)
  {
  }
  
  unsigned color:2;
  unsigned group:6;
} __attribute__((packed));

class Board {
public:
  Cell cells[BIG_N];
  
  Board();

  int color(int p) const { return cells[p].color; }
  
  void move(int x, int y, Color color) { }
};

template<typename T>
class Region {
private:
  class Iterator {
  private:
    const T accept;
    std::bitset<BIG_N> seen;
    byte open[N];
    int size;
  
    void add(int p) {
      if (accept(p) && !seen[p]) {
        seen[p] = true;
        open[size++] = p;
      }
    }

  public:
    Iterator(T accept) : accept(accept), size(0) { }    
    Iterator(int start, T accept): Iterator(accept) { add(start); }

    int operator*() {
      assert(size > 0);
      return open[size - 1];
    }

    void operator++() {
      int p = operator*();
      --size;
      add(p + 1);
      add(p - 1);
      add(p + DELTA_DOWN);
      add(p + DELTA_UP);
    }

    bool operator!=(const Iterator &other) { return size != other.size; }
  };

  const T accept;
  const int start;
      
public:
  Region(int start, T accept) : accept(accept), start(start) {}
  Iterator begin() { return Iterator(start, accept); }
  Iterator end() { return Iterator(accept); }
};

template<typename T> Region<T> makeRegion(int start, T test) { return Region<T>(start, test); }

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

int main() {
  Board b;
  auto region = makeRegion(pos(0, 0), [&b](int p){ return b.color(p) == EMPTY; });
  printf("size %ld %ld\n", sizeof(b), sizeof(region));
  for (int p : region) { printf("%d\n", p); }
}

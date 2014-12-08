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

template<typename T> class Region;

class Board {
public:
  Cell cells[BIG_N];
  
  Board();

  int color(int p) { return cells[p].color; }
  
  template<typename T> Region<T> region(int start, T test);
  
  void move(int x, int y, Color color) { }
};

template<typename T>
class Region {
private:
  const T test;
  Board *board;
  const int start;

  bool accept(int p) { return test(board, p); }
  
  class Iterator {
  private:
    Region *region;
    std::bitset<BIG_N> seen;
    byte open[N];
    int size;
  
    void add(int p) {
      if (region->accept(p) && !seen[p]) {
        seen[p] = true;
        open[size++] = p;
      }
    }

  public:
    Iterator(Region *region, int start): region(region), size(0) {
      add(start);
    }

    Iterator(): region(0), size(0) {}
    
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

    bool operator!=(const Iterator &other) {
      return size != other.size;
    }
  };
      
public:
  Region(T test, Board *board, int start) : test(test), board(board), start(start) {}
  Iterator begin() { return Iterator(this, start); }
  Iterator end() { return Iterator(); }
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

template<typename T>
Region<T> Board::region(int start, T test) { return Region<T>(test, this, start); }

int main() {
  Board b;
  printf("size %ld\n", sizeof(b));
  auto region = b.region(pos(0, 0), [](Board *b, int p){ return b->color(p) == EMPTY; });
  for (int p : region) {
    printf("%d\n", p);
  }
}

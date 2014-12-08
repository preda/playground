#include <stdio.h>
#include <bitset>

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
  C_BLACK = 0,
  C_WHITE = 1,
  C_EMPTY = 2,
  C_BROWN = 3
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
    color(C_EMPTY),
    group(0)
  {
  }
  
  unsigned color:2;
  unsigned group:6;
} __attribute__((packed));

template<typename T> class Walk;

class Board {
public:
  Cell cells[BIG_N];
  
  Board();

  template<typename T> Walk<T> *walk(int start);
  
  void move(int x, int y, Color color) {
  }
};

template<typename T>
class Walk {
private:
  std::bitset<BIG_N> seen;
  byte open[N];
  int size;
  Cell *cells;
  
  void add(int p) {
    if (T::accept(cells[p]) && !seen[p]) {
      seen[p] = true;
      open[size++] = p;
    }
  }
    
public:
  Walk(Board *b, int start) :
    size(0),
    cells(b->cells)
  {
    add(start);
  }

  int pop() {
    if (size <= 0) { return -1; }
    int p = open[--size];
    add(p + 1);
    add(p - 1);
    add(p + DELTA_DOWN);
    add(p + DELTA_UP);
    return p;
  }
};

template<typename T>
Walk<T> *Board::walk(int start) {
  return new Walk<T>(this, start);
}

Board::Board() {
  for (int y = 0; y < SIZE_Y + 1; ++y) {
    cells[y * BIG_X].color = C_BROWN;
    cells[y * BIG_X + SIZE_X + 1].color = C_BROWN;
  }
  for (int x = 0; x < BIG_X; ++x) {
    cells[x].color = C_BROWN;
    cells[(SIZE_Y + 1) * BIG_X + x].color = C_BROWN;
  }
}


#define CONDITION(Name, expr) struct Name { static bool accept(Cell cell) { return expr; } }

CONDITION(White, cell.color == C_WHITE);
CONDITION(Black, cell.color == C_BLACK);
CONDITION(Empty, cell.color == C_EMPTY);

int main() {
  Board b;
  printf("size %ld\n", sizeof(b));
  auto w = b.walk<Empty>(pos(0, 0));
  int p;
  while ((p = w->pop()) >= 0) {
    printf("%d\n", p);
  }
}

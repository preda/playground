#include <stdio.h>
#include <bitset>

typedef unsigned char byte;

using namespace std;

#define DELTAS {DELTA_UP, DELTA_LEFT, DELTA_RIGHT, DELTA_DOWN}

enum Color {
  C_BLACK = 0,
  C_WHITE = 1,
  C_EMPTY = 2,
  C_BROWN = 3
};

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

template<int X, int Y> 
class Board {
private:  
  enum { N = X * Y, BIGN = (X + 2) * (Y + 2) };
  Cell cells[BIGN];

public:
  enum { UP = 0, LEFT = 1, RIGHT = 2, DOWN = 3};
  enum {DELTA_UP = -(X + 2), DELTA_LEFT = -1, DELTA_RIGHT = 1, DELTA_DOWN = -DELTA_UP};
  static const int idx[X * Y];

  static int pos(int y, int x) { return X + 2 + y * (X + 2) + x + 1; }

  template<typename T>
  class Walk {
  private:
    std::bitset<BIGN> seen;
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
      for (int delta : DELTAS) {
        add(p + delta);
      }
      return p;
    }
  };
  
  Board() {
    for (int y = 0; y < Y+1; ++y) {
      cells[y * (X + 2)].color = C_BROWN;
      cells[y * (X + 2) + X + 1].color = C_BROWN;
    }
    for (int x = 0; x < X + 2; ++x) {
      cells[x].color = C_BROWN;
      cells[(Y + 1) * (X + 2) + x].color = C_BROWN;
    }
  }

  template<typename T>
  Walk<T> *walk(int start) {
    return new Walk<T>(this, start);
  }
  
  void move(int x, int y, Color color) {
  }
};

template<> const int Board<6, 6>::idx[] = {
  pos(0, 0), pos(0, 1), pos(0, 2), pos(0, 3), pos(0, 4), pos(0, 5),
  pos(1, 0), pos(1, 1), pos(1, 2), pos(1, 3), pos(1, 4), pos(1, 5),
  pos(2, 0), pos(2, 1), pos(2, 2), pos(2, 3), pos(2, 4), pos(2, 5),
  pos(3, 0), pos(3, 1), pos(3, 2), pos(3, 3), pos(3, 4), pos(3, 5),
  pos(4, 0), pos(4, 1), pos(4, 2), pos(4, 3), pos(4, 4), pos(4, 5),
  pos(5, 0), pos(5, 1), pos(5, 2), pos(5, 3), pos(5, 4), pos(5, 5),
};

#define CONDITION(Name, expr) struct Name { static bool accept(Cell cell) { return expr; } }

CONDITION(White, cell.color == C_WHITE);
CONDITION(Black, cell.color == C_BLACK);
CONDITION(Empty, cell.color == C_EMPTY);

int main() {
  Board<6, 6> b;

  printf("size %ld\n", sizeof(b));
  int start = b.pos(0, 0);
  auto w = b.walk<Empty>(b.pos(0, 0));
  int p;
  while ((p = w->pop()) >= 0) {
    printf("%d\n", p);
  }
}














  
  /*
  int left(int cid) { return cid + DELTA[LEFT]; }
  int right(int cid) { return cid + DELTA[RIGHT]; }
  int up(int cid) { return cid + DELTA[UP]; }
  int down(int cid) { return cid + DELTA[DOWN]; }
  */
// const int Board::DELTAS[] = {DELTA_UP, DELTA_LEFT, DELTA_RIGHT, DELTA_DOWN};

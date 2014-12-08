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

const auto Empty = [](Color color) { return color == EMPTY; };
const auto Black = [](Color color) { return color == BLACK; };

struct Cell {
public:
  Cell(): color(EMPTY), group(0) { }
  Color color:2;
  unsigned group:6;
} __attribute__((packed));


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

class Board {
public:
  Cell cells[BIG_N];

  Board();
  int color(int p) const { return cells[p].color; }
  void move(int p, Color color);

  template<typename T> Region<T> region(int start, T accept);
  template<typename R, typename T> Region2<R, T> border(R subreg, T accept);
  template<typename R> bool isDead(R region) {
    return isEmpty(border(region, Empty));
  }
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
    Color color = cells[p].color;
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
private:
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

template<typename T> Region<T> Board::region(int start, T accept) {
  return Region<T>(this, start, accept);
}

template<typename R, typename T> Region2<R, T> Board::border(R subreg, T accept) {
  return Region2<R, T>(this, subreg, accept);
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
}

void Board::move(int p, Color color) {
  cells[p].color = color;
}

int main() {
  Board b;
  b.move(pos(1, 1), BLACK);
  b.move(pos(0, 1), BLACK);
  auto black = b.region(pos(1, 1), Black);
  print(black);
  auto libs = b.border(black, Empty);
  print(libs);
  printf("Dead %d\n", b.isDead(black));
}

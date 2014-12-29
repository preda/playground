#include "Board.hpp"
#include "TransTable.hpp"
#include <stdio.h>
#include <assert.h>

char *expand(char *line) {
  for (int i = SIZE_X - 1; i >= 0; --i) {
    line[2*i+1] = line[i];
    line[2*i] = ' ';
  }
  line[SIZE_X * 2] = 0;
  return line;
}

char Board::charForPos(int p) {
  return is<BLACK>(p) ? 'x' : is<WHITE>(p) ? 'o' : isEmpty(p) ? '.' : isBorder(p) ? '-' : '?';
}

int Board::groupColor(int gid) {
  for (int p = 0; p < BIG_N; ++p) {
    if (gids[p] == gid && (is<BLACK>(p) || is<WHITE>(p))) {
      return is<BLACK>(p) ? BLACK : WHITE;
    }
  }
  printf("groupColor gid %d %Lx %d\n", gid, groups[gid], gids[P(0, 0)]);
  assert(false);
}

void Board::print(uint64_t pointsBlack, uint64_t pointsWhite) {
  char line1[256], line2[256], line3[256];
  for (int y = 0; y < SIZE_Y; ++y) {
    for (int x = 0; x < SIZE_X; ++x) {
      int p = P(y, x);
      line1[x] = charForPos(p);
      line2[x] = '0' + gids[p];
      bool isPointBlack = IS(p, pointsBlack);
      bool isPointWhite = IS(p, pointsWhite);
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

template<int C> void doPlay(Board &board, int p) {
  if (board.isSuicide<C>(p)) {
    printf("suicide\n");
    return;
  }
  
  board.play<C>(p);
  uint64_t pointsMe = board.bensonAlive<C>();
  uint64_t pointsOth = board.bensonAlive<1-C>();
  if (C == BLACK) {
    board.print(pointsMe, pointsOth);
  } else {
    board.print(pointsOth, pointsMe);
  }  
}

static bool isValid(int y, int x) { return y >= 0 && y < SIZE_Y && x >= 0 && x < SIZE_X; }

int main() {
  TransTable tt;
  
  Board board;
  uint64_t pointsMe = 0, pointsOth = 0;
  board.print(pointsMe, pointsOth);
  while (true) {
    char buf[16] = {0};
    int y = -1;
    int x = -1;
    printf("> ");
    if (scanf("%1s %1d %1d", buf, &y, &x) != 3) { continue; }
    char c = buf[0];
    int col = c == 'b' ? BLACK : c == 'w' ? WHITE : EMPTY;
    if (isBlackOrWhite(col) && isValid(y, x)) {
      int p = P(y, x);
      if (board.isEmpty(p)) {
        if (col == BLACK) {
          doPlay<BLACK>(board, p);
        } else {
          doPlay<WHITE>(board, p);
        }
      }
    }
  }
}

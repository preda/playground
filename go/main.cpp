#include "Board.hpp"
#include <stdio.h>
#include <assert.h>

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
  Board board;
  Bitset pointsMe, pointsOth;
  board.print(pointsMe, pointsOth);
  while (true) {
    char buf[16] = {0};
    int y = -1;
    int x = -1;
    printf("> ");
    if (scanf("%1s %1d %1d", buf, &y, &x) != 3) { continue; }
    char c = buf[0];
    int col = c == 'b' ? BLACK : c == 'w' ? WHITE : EMPTY;
    if (isBlackOrWhite(col) && isValid(y, x) && board.color(pos(y, x)) == EMPTY) {
      if (!board.play(pos(y, x), col)) {
        printf("suicide\n");
      }

      unsigned aliveGroupBitsMe, aliveGroupBitsOth;
      board.bensonAlive(col, pointsMe, &aliveGroupBitsMe);
      board.bensonAlive((1-col), pointsOth, &aliveGroupBitsOth);
      if (col == BLACK) {
        board.print(pointsMe, pointsOth);
      } else {
        board.print(pointsOth, pointsMe);
      }
    }
  }
}

#include "Node.hpp"
#include <stdio.h>
#include <assert.h>

template<int C> void doPlay(Node &node, int p) {
  if (node.isSuicide<C>(p)) {
    printf("suicide\n");
    return;
  }
  
  node = node.play<C>(p);
  node.print();
}

static bool isValid(int y, int x) { return y >= 0 && y < SIZE_Y && x >= 0 && x < SIZE_X; }

int main() {  
  Node node;
  node.print();
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
      if (node.isEmpty(p)) {
        if (col == BLACK) {
          doPlay<BLACK>(node, p);
        } else {
          doPlay<WHITE>(node, p);
        }
      }
    }
  }
}

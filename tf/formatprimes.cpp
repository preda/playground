#include <stdio.h>

int main() {
  int d;
  int cnt = 10;
  while (scanf("%d ", &d)) {
    printf("%5d, ", d);
    if (d == 65521) { break; }
    if (!--cnt) {
      printf("\n");
      cnt = 10;
    }
  }
}

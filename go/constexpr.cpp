#include "go.hpp"
#include <stdio.h>

uint64_t borderPoints() {
  uint64_t points = 0;
  for (int y = 0; y < SIZE_Y; ++y) {
    SET(P(y, -1), points);
    SET(P(y, SIZE_X), points);
  }
  for (int x = 0; x < BIG_X; ++x) {
    SET(P(-1, x), points);
    SET(P(SIZE_Y, x), points);
  }
  return points;
}

template<typename T> static uint64_t selectPoints(T t) {
  uint64_t points = 0;
  for (int y = 0; y < SIZE_Y; ++y) {
    for (int x = 0; x < SIZE_X; ++x) {
      if (t(y, x)) { SET(P(y, x), points); }
    }
  }
  return points;
}

int main() {
  printf("\
#define BORDER 0x%lx\n\
#define INSIDE 0x%lx\n\
#define HALF_Y 0x%lx\n\
#define HALF_X 0x%lx\n\
#define HALF_DIAG 0x%lx\n",
         borderPoints(),
         selectPoints([](int y, int x) {return true; }),
         selectPoints([](int y, int x) {return y < (SIZE_Y + 1) / 2; }),
         selectPoints([](int y, int x) {return x < (SIZE_X + 1) / 2; }),
         selectPoints([](int y, int x) {return x >= y; }));
}

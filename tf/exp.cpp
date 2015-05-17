#include <stdio.h>

typedef unsigned long long u64;
typedef __uint128_t u128;

// #define N (4 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23)
// #define N (4 * 3 * 5 * 7 * 11 * 13 * 17 * 19)
#define NCLASS (4 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23)
// * 31 * 37 * 41 * 43 * 47)

// whether 2 * k * p + 1 == 1 or 7 mod 8.
extern inline bool q1or7mod8(unsigned p, u64 k) {
  return !(k & 3) || ((k & 3) + (p & 3) == 4);
}

extern inline bool ok(unsigned p, unsigned k, unsigned prime) {
  return (((p % prime) * 2) * (u64)k + 1) % prime != 0;
}

static /*__attribute__((noinline))*/ bool acceptClass(unsigned p, unsigned k) {
  return q1or7mod8(p, k) && ok(p, k, 3) && ok(p, k, 5) && ok(p, k, 7)
    && ok(p, k, 11) && ok(p, k, 13) && ok(p, k, 17) && ok(p, k, 19) && ok(p, k, 23);
  // && ok(p, k, 29) && ok(p, k, 31);
  // && ok(p, k, 31) && ok(p, k, 37) && ok(p, k, 41) && ok(p, k, 43) && ok(p, k, 47);
    
}

int main() {
  unsigned p = 119904229;
  u64 k = (((u128) 1) << (65 - 1)) / p;
  k -= k % NCLASS;
  printf("start K %llu\n", k);
  int n = 0;
  // int end = 100000000;
  for (int c = 0; c < NCLASS; ++c) {
    if (acceptClass(p, c)) { ++n; }
  }
  printf("%d/%d (%f)\n", n, NCLASS, (double)n * 100 / NCLASS);
}


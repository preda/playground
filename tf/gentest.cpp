#include <stdio.h>
#include <vector>
#include <algorithm>
#include <assert.h>

#define N 2868

unsigned ex[N];
unsigned bit_min[N];
unsigned long long k[N];

typedef __uint128_t u128;

struct Test {
  unsigned ex;
  u128 m;
  unsigned long long k;
};

bool less(Test a, Test b) { return a.m < b.m; }

int main() {
#define exp ex
#include "selfraw.c"
  
  std::vector<Test> v;
  
  for (int i = 0; i < N; ++i) {
    unsigned e = ex[i];
    unsigned long long kk = k[i];
    u128 m = kk * (u128) e * 2;
    /*
    float bit = log2(((double)kk) * 2 * e);
    if (bit_min[i] != (int)bit) {
      printf("%d: %u %f\n", i, bit_min[i], bit);
    }
    assert(bit_min[i] == (int) bit);
    */
    if ((m>>64) && !(m>>76) && !(e & 0x80000000)) {
      v.push_back(Test{e, m, kk});
    }
  }
  std::sort(v.begin(), v.end(), less);
  printf("Test tests[] = {\n");
  for (Test t : v) {
    printf("{%10u, %18lluull},\n", t.ex, t.k);
  }
  printf("};\n");
}

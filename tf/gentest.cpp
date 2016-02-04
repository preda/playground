#include <stdio.h>

#define N 2867

unsigned exp[N];
unsigned bit_min[N];
unsigned long long k[N];

int main() {
#include "selfraw.c"
  printf("Test tests[] = {\n");
  for (int i = 0; i < N; ++i) {
    if (bit_min[i] >= 64 && bit_min[i] < 76 && !(exp[i] & 0x80000000)) {
      printf("{%10u, %22lluull},\n", exp[i], k[i]);
    }
  }
  printf("};\n");
}

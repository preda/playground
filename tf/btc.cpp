#include <stdio.h>
#include "common.h"

u16 modInv16(u64 step, u16 prime) {
  u16 n = step % prime;
  u16 q = prime / n;
  u16 d = prime - q * n;
  int x = -q;
  int prevX = 1;
  while (d) {
    q = n / d;
    { u16 save = d; d = n - q * d; n = save;         }
    { int save = x; x = prevX - q * x; prevX = save; }
  }
  return (prevX >= 0) ? prevX : (prevX + prime);
}

u16 bitToClear(u32 exp, u64 k, u16 prime, u16 inv) {
  // 2 * exp * k + 1
  // unsigned qinv = (unsigned) (q * inv % prime);
  // return qinv ? prime - qinv : 0;
  
  u16 kmod = k % prime;
  u16 qmod = ((exp << 1) * (u64) kmod + 1) % prime;
  return (prime - qmod) * inv % prime;
}

// #define NCLASS (4 * 3 * 5 * 7)

u16 bitToClear420(u32 exp, u64 k, u16 prime) {
  u64 step = 2 * 420 * (u64) exp;  
  u16 inv = modInv16(step, prime);
  // assert(inv == modInv32(step, prime));
  return bitToClear(exp, k, prime, inv);
}

u16 bitToClear1(u32 exp, u64 k, u16 prime) {
  u64 step = 2 * 1 * (u64) exp;  
  u16 inv = modInv16(step, prime);
  //assert(inv == modInv32(step, prime));
  return bitToClear(exp, k, prime, inv);
}

u16 base(u32 exp, u16 prime) {
  return bitToClear1(exp, 0, prime);
}

u16 bump(u16 v, int n, u16 prime) {
  int x = v - n % prime;
  return x >= 0 ? x : (x + prime);
}

u16 btcFun(u32 exp, u64 k, u16 prime) {
  int x = base(exp, prime) - (k * 420) % prime;
  return x >= 0 ? x : x + prime;
}

int main() {
  u32 exp = 119904229;
  u64 k = 123125643;
  u16 prime = 127;
  int delta = 373232;

  u16 btc = bitToClear420(exp, k, prime);
  printf("%d %d %d\n", btc, bitToClear420(exp, k + (delta * 420), prime), bump(btc, delta, prime));
}

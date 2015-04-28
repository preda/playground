// Copyright (c) Mihai Preda, 2015.

#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

typedef unsigned long long u64;
typedef __uint128_t u128;

struct U3 { unsigned a, b, c; };
struct U4 { unsigned a, b, c, d; };
struct U6 { unsigned a, b, c, d, e, f; };

#define __both__ __device__ __host__

static __both__ void print(U3 a) {
  printf("0x%08x%08x%08x\n", a.c, a.b, a.a);
}

static __both__ void print(U6 a) {
  printf("0x%08x%08x%08x'%08x%08x%08x\n", a.f, a.e, a.d, a.c, a.b, a.a);
}

/*
__device__ static void printD(U3 a) {
  print(a);
}

__device__ static void printD(U6 a) {
  print(a);
}
*/

// --- MUL ---

__device__ static unsigned mul(unsigned a, unsigned b) { return a * b; }

/*
__device__ static unsigned madhi_cc(unsigned a, unsigned b, unsigned c) {
  unsigned r;
  asm("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ static unsigned mulhic(unsigned a, unsigned b) {
  unsigned r;
  asm("madc.hi.u32 %0, %1, %2, 0;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ static unsigned madhic_cc(unsigned a, unsigned b, unsigned c) {
  unsigned r;
  asm("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}
*/

// --- ADD / SUB ---

__device__ static unsigned add_cc(unsigned a, unsigned b) {
  unsigned r;
  asm("add.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b));
  return r;
}

__device__ static unsigned addc(unsigned a, unsigned b) {
  unsigned r;
  asm("addc.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b));
  return r;
}

__device__ static unsigned addc_cc(unsigned a, unsigned b) {
  unsigned r;
  asm("addc.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b));
  return r;
}

__device__ static unsigned sub_cc(unsigned a, unsigned b) {
  unsigned r;
  asm("sub.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b));
  return r;
}

__device__ static unsigned subc(unsigned a, unsigned b) {
  unsigned r;
  asm("subc.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b));
  return r;
}

__device__ static unsigned subc_cc(unsigned a, unsigned b) {
  unsigned r;
  asm("subc.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b));
  return r;
}


// --- Shift ---

__device__ static unsigned shl(unsigned a, unsigned b, int n) {
  unsigned r;
  asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(n));
  return r;
}

__device__ static unsigned shr(unsigned a, unsigned b, int n) {
  unsigned r;
  asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(n));
  return r;
}


// --- Multiprecision ---


__device__ static U3 add(U3 x, U3 y) {
  unsigned a =  add_cc(x.a, y.a);
  unsigned b = addc_cc(x.b, y.b);
  unsigned c = addc_cc(x.c, y.c);
  assert(!addc(0, 0));
  return {a, b, c};
}

__device__ static U6 add(U6 x, U6 y) {
  unsigned a =  add_cc(x.a, y.a);
  unsigned b = addc_cc(x.b, y.b);
  unsigned c = addc_cc(x.c, y.c);
  unsigned d = addc_cc(x.d, y.d);
  unsigned e = addc_cc(x.e, y.e);
  unsigned f = addc_cc(x.f, y.f);
  assert(!addc(0, 0));
  return {a, b, c, d, e, f};
}

__device__ static U6 sub(U6 x, U6 y) {
  unsigned a =  sub_cc(x.a, y.a);
  unsigned b = subc_cc(x.b, y.b);
  unsigned c = subc_cc(x.c, y.c);
  unsigned d = subc_cc(x.d, y.d);
  unsigned e = subc_cc(x.e, y.e);
  unsigned f = subc_cc(x.f, y.f);
  assert(!subc(0, 0));
  return {a, b, c, d, e, f};
}

__device__ static U6 subshl(U6 x, U6 y, int n) {
  unsigned a =  sub_cc(x.a, y.a << n);
  unsigned b = subc_cc(x.b, shl(y.a, y.b, n));
  unsigned c = subc_cc(x.c, shl(y.b, y.c, n));
  unsigned d = subc_cc(x.d, shl(y.c, y.d, n));
  unsigned e = subc_cc(x.e, shl(y.d, y.e, n));
  unsigned f =    subc(x.f, shl(y.e, y.f, n));
  return {a, b, c, d, e, f};
}

__device__ static U4 mul(U3 v, unsigned n) {
  unsigned a =         mul(n, v.a);
  unsigned b =  add_cc(mulhi(n, v.a), mul(n, v.b));
  unsigned c = addc_cc(mulhi(n, v.b), mul(n, v.c));
  unsigned d =    addc(mulhi(n, v.c), 0);
  return {a, b, c, d};

  /* equivalent alternative:
  out[0] = mul(n, d0);
  out[1] = madhi_cc(n, d0, mul(n, d1));
  out[2] = madhic_cc(n, d1, mul(n, d2));
  out[3] = mulhic(n, d2);
  */
}

__device__ static U3 mul_lo(U3 x, U3 y) {
  unsigned a, b, c;
  asm("{\n"
      "mul.lo.u32    %0, %3, %6;\n"
      "mul.hi.u32    %1, %3, %6;\n"
      "mad.lo.cc.u32 %1, %3, %7, %1;\n"
      "mul.lo.u32    %2, %3, %8;\n"
      "madc.hi.u32   %2, %3, %7, %2;\n"
      "mad.lo.cc.u32 %1, %4, %6, %1;\n"
      "madc.hi.u32   %2, %4, %6, %2;\n"
      "mad.lo.u32    %2, %4, %7, %2;\n"
      "mad.lo.u32    %2, %5, %6, %2;\n"
      "}\n" : "=r"(a), "=r"(b), "=r"(c) : "r"(x.a), "r"(x.b), "r"(x.c), "r"(y.a), "r"(y.b), "r"(y.c));
  return {a, b, c};
}

// Inspired my mfaktc's square96 implem.
__device__ static U6 square(U3 x) {
  unsigned a, b, c, d, e, f;
  asm("{\n"
      "mul.lo.u32      %0, %6, %6;     \n"  // (d0 * d0).lo
      "mul.lo.u32      %1, %6, %7;     \n"  // (d0 * d1).lo
      "mul.hi.u32      %2, %6, %7;     \n"  // (d0 * d1).hi      
      "add.cc.u32      %1, %1, %1;     \n"  // 2 * (d0 * d1).lo
      "addc.cc.u32     %2, %2, %2;     \n"  // 2 * (d0 * d1).hi
      "madc.hi.cc.u32  %3, %7, %7, 0;  \n"  // (d1 * d1).hi
      "madc.lo.u32     %4, %8, %8, 0;  \n"  // (d2 * d2).lo; %4 <= 0xFFFFFFFA => no carry to %5 needed!
      "add.u32         %5, %8, %8;     \n"  // 2 * d2; d2 < 2**31
      "mad.hi.cc.u32   %1, %6, %6, %1; \n"  // (d0 * d0).hi
      "madc.lo.cc.u32  %2, %7, %7, %2; \n"  // (d1 * d1).lo
      "madc.lo.cc.u32  %3, %7, %5, %3; \n"  // 2 * (a.d1 * a.d2).lo
      "addc.u32        %4, %4, 0;      \n"  // %4 <= 0xFFFFFFFB => not carry to %5 needed
      "mad.lo.cc.u32   %2, %6, %5, %2; \n"  // 2 * (d0 * d2).lo
      "madc.hi.cc.u32  %3, %6, %5, %3; \n"  // 2 * (d0 * d2).hi
      "madc.hi.cc.u32  %4, %7, %5, %4; \n"  // 2 * (d1 * d2).hi
      "madc.hi.u32     %5, %8, %8, 0;  \n"  // (d2 * d2).hi
      "}\n" : "=r"(a), "=r"(b), "=r"(c), "=r"(d), "=r"(e), "=r"(f) : "r"(x.a), "r"(x.b), "r"(x.c));
  return {a, b, c, d, e, f};
}


__device__ static U6 shl1w(U4 x)  { return {0, x.a, x.b, x.c, x.d, 0}; }
__device__ static U6 shl2w(U4 x)  { return {0, 0, x.a, x.b, x.c, x.d}; }
__device__ static U6 shl3w(U3 x)  { return {0, 0, 0, x.a, x.b, x.c}; }
__device__ static U3 shr1w(U4 x)  { return {x.b, x.c, x.d}; }
__device__ static U6 makeU6(U3 x) { return {x.a, x.b, x.c, 0, 0, 0}; }
__device__ static U6 makeU6(U4 x) { return {x.a, x.b, x.c, x.d, 0, 0}; }


__device__ static U3 shl(U3 x, int n) {
  return {x.a << n, shl(x.a, x.b, n), shl(x.b, x.c, n)};
}

__device__ static U6 shl(U6 x, int n) {
  return {x.a << n, shl(x.a, x.b, n), shl(x.b, x.c, n), shl(x.c, x.d, n), shl(x.d, x.e, n), shl(x.e, x.f, n)};
}

__device__ static U3 shr(U3 x, int n) {
  return {shr(x.a, x.b, n), shr(x.b, x.c, n), x.c >> n};
}

/*
__device__ static U3 avg(U3 x, U3 y) {
  U4 z = x + y;
  return {shr(a, b, 1), shr(b, c, 1), shr(c, d, 1)};
}
*/


// --- MOD ---


// 3W = 6W % 3W; y >= 2**95
__device__ static U3 mod(U6 x, U3 y, unsigned R) {
  print(x);
  print(y);
  printf("R %x\n", R);
  
  assert(y.c & 0x80000000);
  
  x = subshl(x, shl2w(mul(y, mulhi(x.f, R))), 1);
  print(x);
  assert((x.f & 0xfffffff0) == 0);
  x = subshl(x, shl1w(mul(y, mulhi(shl(x.e, x.f, 28), R))), 5);
  print(x);
  assert(x.f == 0 && (x.e & 0xffffff00) == 0);
  x = subshl(x, makeU6(mul(y, mulhi(shl(x.d, x.e, 24), R))), 9);
  print(x);
  assert(x.f == 0 && x.e == 0 && (x.d & 0xfffff000) == 0);
  unsigned q = mulhi(shl(x.c, x.d, 20), R) >> 19;
  U6 yq = makeU6(mul(y, q));
  print(yq);
  printf("q %x\n", q);
  x = sub(x, yq);
  print(x);
  // printf("%x %x %x\n", x.d, x.e, x.f);
  assert(!x.f && !x.e && !x.d);
  
  return {x.a, x.b, x.c};
}

// y > 2**64
__device__ static U3 mod(U6 x, U3 y) {
  assert(y.c);
  int shift = __clz(y.c);
  assert(__clz(x.f) >= shift);
  x = shl(x, shift);
  y = shl(y, shift);
  
  unsigned R = 0xffffffffffffffffULL / ((0x100000000ULL | shl(y.b, y.c, 1)) + 1);  
  return shr(mod(x, y, R), shift);
}

// find mp such that (mp * m) mod 2^96 = 2^96 - 1
__device__ static U3 mprime(U3 m) {
  U3 u{1, 0, 0};
  U3 v{0, 0, 0};
  m = add(shr(m, 1), (U3){1, 0, 0});
  
  for (int i = 96; i; --i) {
    bool odd = u.a & 1;
    u = shr(u, 1);
    v = shr(v, 1);
    if (odd) {
      u = add(u, m);
      v.c |= 0x80000000;
    }
  }
  return v;
}

__device__ static unsigned mprime0(U3 m) {
  unsigned u = 1;
  unsigned v = 0;
  unsigned m0 = shr(m.a, m.b, 1);
  m0 += 1;
  for (int i = 32; i; --i) {
    bool odd = u & 1;
    u >>= 1;
    v >>= 1;
    if (odd) {
      u += m0;
      v |= 0x80000000;
    }
  }
  return v;
}

/*
// find u, v such that (u << 96) - v * b == 1
__device__ static void gcd(U3 b, U3 *pu, U3 *pv) {
  U3 u{1, 0, 0};
  U3 v{0, 0, 0};
  for (int i = 96; i; --i) {
    v = shr(v, 1);
    if (u.a & 1) {
      u = avg(u, b);
      v.c |= 0x80000000;
    } else {
      u = shr(u, 1);
    }
  }
  *pu = u;
  *pv = v;
}
*/

// Montgomery Reduction
// See https://www.cosic.esat.kuleuven.be/publications/article-144.pdf
// Returns x * U^-1 mod m
__device__ static U3 montRed(U6 x, U3 m, unsigned mp0) {
  unsigned t = x.a * mp0;
  U4 f = mul(m, t);
  x = add(x, makeU6(f));
  assert(!x.a);
  t = x.b * mp0;
  f = mul(m, t);
  x = add(x, shl1w(f));
  assert(!x.b);
  t = x.c * mp0;
  f = mul(m, t);
  x = add(x, shl2w(f));
  assert(!x.a && !x.b && !x.c);
  return {x.d, x.e, x.f};
}

__global__ void test1() {
  U3 m = {0xffffffff, 0xffffffff, 0xffffffff};
  unsigned mp = mprime0(m);
  // U3 u = mod((U6){0, 0, 0, 1, 0, 0}, m);

  U3 aa = {42, 0, 0};
  U3 a = mod(shl3w(aa), m);
  U6 a2 = square(a);
  U3 c = montRed(a2, m, mp);
  U3 cc = montRed(makeU6(c), m, mp); 
  print(a);
  print(c);
  print(cc);
  
  /*
  U3 mp = mprime(m);
  unsigned mp0 = mprime0(m);
  printD(m);
  printD(mp);
  printD(mul_lo(m, mp));
  printf("%08x %08x\n", mp0, mp0 * m.a);
  */
}

#define N 32

__managed__ U6 as[N];
__managed__ U3 bs[N];
__managed__ U3 out[N * N];

int main() {
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  // U3 b{0x80ffffff, 0xfffff345, 0x0};
  test1<<<1, 1>>>();
  cudaDeviceSynchronize();

  // print(b);
  // print(as[0]);
  /*
  test2<<<1, 1>>>(out, as, bs);
  cudaDeviceSynchronize();
  print(as[0]);
  */
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

/*
void gcdHost(u64 b, u64 *pu, u64 *pv) {
  u64 u = 1, v = 0;
  for (int i = 64; i; --i) {
    v >>= 1;
    if (u & 1) {
      u = (u + b) >> 1;
      v |= (1ull << 63);
    } else {
      u >>= 1;
    }
  }
  *pu = u;
  *pv = v;
}

unsigned random32() {
  return (((unsigned)random()) << 1) | (random() & 1);
}

*/

  /*
  int x = threadIdx.x;
  int y = threadIdx.y;
  N192 a = as[x];
  N96 b = bs[y];
  out[x + y * blockDim.y] = mod(a, b);
  */

  /*
  for (int i = 0; i < N; ++i) {
    N192 a = random192();
    a.d5 |= 0x80000000;
    as[i] = a;
  }
  for (int i = 0; i < N; ++i) {
    N96 b = random96();
    b.d2 |= 0x80000000;
    bs[i] = b;
  }
  printf("Done rnd\n");
  dim3 blockDim(N, N);
  */


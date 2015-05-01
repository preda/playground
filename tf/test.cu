// Copyright (c) Mihai Preda, 2015.

#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

// #define assert(x) 

typedef unsigned long long u64;
typedef __uint128_t u128;

struct U3 { unsigned a, b, c; };
struct U4 { unsigned a, b, c, d; };
struct U5 { unsigned a, b, c, d, e; };
struct U6 { unsigned a, b, c, d, e, f; };

#define __both__ __device__ __host__

static __both__ void print(U3 a) {
  printf("0x%08x%08x%08x\n", a.c, a.b, a.a);
}

static __both__ void print(U4 a) {
  printf("0x%08x%08x%08x%08x\n", a.d, a.c, a.b, a.a);
}

static __both__ void print(U6 a) {
  printf("0x%08x%08x%08x%08x%08x%08x\n", a.f, a.e, a.d, a.c, a.b, a.a);
}

// #define print(x)

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

__device__ static U6 add(U6 x, U6 y) {
  unsigned a, b, c, d, e, f, carryOut;
  asm("add.cc.u32  %0, %7,  %13;"
      "addc.cc.u32 %1, %8,  %14;"
      "addc.cc.u32 %2, %9,  %15;"
      "addc.cc.u32 %3, %10, %16;"
      "addc.cc.u32 %4, %11, %17;"
      "addc.cc.u32 %5, %12, %18;"
      "addc.u32    %6, 0, 0;"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d), "=r"(e), "=r"(f), "=r"(carryOut)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(x.d), "r"(x.e), "r"(x.f),
        "r"(y.a), "r"(y.b), "r"(y.c), "r"(y.d), "r"(y.e), "r"(y.f));
  assert(!carryOut);
  return {a, b, c, d, e, f};
}

__device__ static U4 sub(U4 x, U4 y) {
  unsigned a, b, c, d, carryOut;
  asm("sub.cc.u32  %0, %5,  %9;"
      "subc.cc.u32 %1, %6,  %10;"
      "subc.cc.u32 %2, %7,  %11;"
      "subc.cc.u32 %3, %8,  %12;"
      "subc.u32    %4, 0, 0;"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d), "=r"(carryOut)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(x.d),
        "r"(y.a), "r"(y.b), "r"(y.c), "r"(y.d));
  assert(!carryOut);
  return {a, b, c, d};
}

__device__ static U5 sub(U5 x, U5 y) {
  unsigned a, b, c, d, e, carryOut;
  asm("sub.cc.u32  %0, %6,   %11;"
      "subc.cc.u32 %1, %7,   %12;"
      "subc.cc.u32 %2, %8,   %13;"
      "subc.cc.u32 %3, %9,   %14;"
      "subc.cc.u32 %4, %10,  %15;"
      "subc.u32    %5, 0, 0;"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d), "=r"(e), "=r"(carryOut)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(x.d), "r"(x.e),
        "r"(y.a), "r"(y.b), "r"(y.c), "r"(y.d), "r"(y.e));
  assert(!carryOut);
  return {a, b, c, d, e};
}

/*
__device__ static U6 subShl2w(U6 x, U4 y) {
  U4 r = sub((U4){x.c, x.d, x.e, x.f}, y);
  return {x.a, x.b, r.a, r.b, r.c, r.d};
}

__device__ static U6 sub(U6 x, U6 y) {
  unsigned a, b, c, d, e, f, carryOut;
  asm("sub.cc.u32  %0, 1, 0;"
      "sub.cc.u32  %0, %7,  %13;"
      "subc.cc.u32 %1, %8,  %14;"
      "subc.cc.u32 %2, %9,  %15;"
      "subc.cc.u32 %3, %10, %16;"
      "subc.cc.u32 %4, %11, %17;"
      "subc.cc.u32 %5, %12, %18;"
      "subc.u32    %6, 0, 0;"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d), "=r"(e), "=r"(f), "=r"(carryOut)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(x.d), "r"(x.e), "r"(x.f),
        "r"(y.a), "r"(y.b), "r"(y.c), "r"(y.d), "r"(y.e), "r"(y.f));
  assert(!carryOut);
  return {a, b, c, d, e, f};
}
*/

__device__ static U4 mul(U3 x, unsigned n) {
  unsigned a, b, c, d;
  asm("mul.lo.u32     %0, %4, %7;"
      "mul.hi.u32     %1, %4, %7;"
      "mad.lo.cc.u32  %1, %5, %7, %1;"
      "mul.hi.u32     %2, %5, %7;"
      "madc.lo.cc.u32 %2, %6, %7, %2;"
      "madc.hi.u32    %3, %6, %7, 0;"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(n));
  return {a, b, c, d};
}

// Inspired my mfaktc's square96 implem.
__device__ static U6 square(U3 x) {
  unsigned a, b, c, d, e, f;
  asm("mul.lo.u32      %0, %6, %6;"      // (d0 * d0).lo
      "mul.lo.u32      %1, %6, %7;"      // (d0 * d1).lo
      "mul.hi.u32      %2, %6, %7;"      // (d0 * d1).hi      
      "add.cc.u32      %1, %1, %1;"      // 2 * (d0 * d1).lo
      "addc.cc.u32     %2, %2, %2;"      // 2 * (d0 * d1).hi
      "madc.hi.cc.u32  %3, %7, %7, 0;"   // (d1 * d1).hi
      "madc.lo.u32     %4, %8, %8, 0;"   // (d2 * d2).lo; %4 <= 0xFFFFFFFA => no carry to %5 needed!
      "add.u32         %5, %8, %8;"      // 2 * d2; d2 < 2**31
      "mad.hi.cc.u32   %1, %6, %6, %1;"  // (d0 * d0).hi
      "madc.lo.cc.u32  %2, %7, %7, %2;"  // (d1 * d1).lo
      "madc.lo.cc.u32  %3, %7, %5, %3;"  // 2 * (a.d1 * a.d2).lo
      "addc.u32        %4, %4, 0;"       // %4 <= 0xFFFFFFFB => not carry to %5 needed
      "mad.lo.cc.u32   %2, %6, %5, %2;"  // 2 * (d0 * d2).lo
      "madc.hi.cc.u32  %3, %6, %5, %3;"  // 2 * (d0 * d2).hi
      "madc.hi.cc.u32  %4, %7, %5, %4;"  // 2 * (d1 * d2).hi
      "madc.hi.u32     %5, %8, %8, 0;"   // (d2 * d2).hi
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d), "=r"(e), "=r"(f)
      : "r"(x.a), "r"(x.b), "r"(x.c));
  U6 r{a, b, c, d, e, f};
  return r;
}

// __device__ static U6 shl1w(U4 x)  { return {0, x.a, x.b, x.c, x.d, 0}; }
__device__ static U5 shl1w(U4 x)  { return {0, x.a, x.b, x.c, x.d}; }
__device__ static U6 shl2w(U4 x)  { return {0, 0, x.a, x.b, x.c, x.d}; }
__device__ static U6 makeU6(U3 x) { return {x.a, x.b, x.c, 0, 0, 0}; }
__device__ static U6 makeU6(U4 x) { return {x.a, x.b, x.c, x.d, 0, 0}; }
__device__ static U6 makeU6(U5 x) { return {x.a, x.b, x.c, x.d, x.e, 0}; }
__device__ static U5 makeU5(U4 x) { return {x.a, x.b, x.c, x.d, 0}; }
__device__ static U4 makeU4(U3 x) { return {x.a, x.b, x.c, 0}; }

__host__ static U3 makeU3(u128 x) {
  assert(!(unsigned) (x >> 96));
  return { (unsigned) x, (unsigned) (x >> 32), (unsigned) (x >> 64)};
}

__device__ static U3 shl(U3 x, int n) {
  assert(n >= 0 && n < 32 && !(x.c >> (32 - n)));
  return {x.a << n, shl(x.a, x.b, n), shl(x.b, x.c, n)};
}

__device__ static U4 shl(U4 x, int n) {
  // printf("shl %d ", n); print(x);
  assert(n >= 0 && n < 32 && !(x.d >> (32 - n)));
  return {x.a << n, shl(x.a, x.b, n), shl(x.b, x.c, n), shl(x.c, x.d, n)};
}

__device__ static U5 shl(U5 x, int n) {
  assert(n >= 0 && n < 32 && !(x.e >> (32 - n)));
  return {x.a << n, shl(x.a, x.b, n), shl(x.b, x.c, n), shl(x.c, x.d, n), shl(x.d, x.e, n)};
}

__device__ static U6 shl(U6 x, int n) {
  assert(n >= 0 && n < 32 && !(x.f >> (32 - n)));
  return {x.a << n, shl(x.a, x.b, n), shl(x.b, x.c, n), shl(x.c, x.d, n), shl(x.d, x.e, n), shl(x.e, x.f, n)};
}

/*
__device__ static U6 shl(U6 x, int n) {
  assert(n >= 0 && n < 32 && !(x.f >> (32 - n)));
  return {x.a << n, shl(x.a, x.b, n), shl(x.b, x.c, n), shl(x.c, x.d, n), shl(x.d, x.e, n), shl(x.e, x.f, n)};
}
*/

__device__ static U3 shr(U3 x, int n) {
  assert(n >= 0 && n < 32 && !(x.a << (32 - n)));
  return {shr(x.a, x.b, n), shr(x.b, x.c, n), x.c >> n};
}

// m >= 2^93 && m < 2^94.
__device__ __noinline__ U3 mod(U4 xx, U3 m, int shift, unsigned R) {
  unsigned n;
  assert((m.c >> 29) == 1);
  U5 x = makeU5(xx);
  x = shl(x, shift);
  
  n = mulhi(x.e, R);  
  x = sub(x, shl(shl1w(mul(m, n)), 3));
  assert(!(x.e & 0xfffffff0));
  
  n = mulhi(shl(x.d, x.e, 28), R);
  x = sub(x, shl(makeU5(mul(m, n)), 7));
  assert(!x.e && !(x.d & 0xffffff00));

  n = mulhi(shl(x.c, x.d, 24), R) >> 21;
  x = sub(x, makeU5(mul(m, n)));
  assert(!x.e && !x.d);
  
  /*
  n = mulhi(x.d, R);  
  x = sub(x, shl(mul(m, n), 3));
  assert(!(x.d & 0xfffffff0));
  n = mulhi(shl(x.c, x.d, 28), R) >> 25;
  x = sub(x, mul(m, n));
  assert(!x.d);
  */
  
  return shr((U3) {x.a, x.b, x.c}, shift);
}

// Compute mp0 such that: (unsigned) (m * mp0) == 0xffffffff; using variant extended euclidian algorithm.
__device__ static unsigned mprime0(U3 m) {
  unsigned m0 = shr(m.a, m.b, 1) + 1;
  unsigned u = m0;
  unsigned v = 0x80000000;
  #pragma unroll 1
  for (int i = 0; i < 31; ++i) {
    v = shr(v, u, 1);
    u = (u >> 1) + ((u & 1) ? m0 : 0);
  }
  return v;
}

// Montgomery Reduction
// See https://www.cosic.esat.kuleuven.be/publications/article-144.pdf
// Returns x * U**-1 mod m
__device__ static U3 montRed(U6 x, U3 m, unsigned mp0) {
  unsigned t = x.a * mp0;
  U4 f = mul(m, t);
  x = add(x, makeU6(f));
  assert(!x.a);
  t = x.b * mp0;
  f = mul(m, t);
  x = add(x, makeU6(shl1w(f)));
  assert(!x.b);
  t = x.c * mp0;
  f = mul(m, t);
  x = add(x, shl2w(f));
  assert(!x.a && !x.b && !x.c);
  U3 r{x.d, x.e, x.f};
  return r;
}

// return a**32 in montgomery space modulo m.
__device__ static U3 power32(U3 a, U3 m, unsigned mp0) {
  for (int i = 0; i < 5; ++i) {
    a = montRed(square(a), m, mp0);
    // print(montRed(makeU6(a), m, mp0));
  }
  return a;
}

// return x < y;
__device__ static bool less(U3 x, U3 y) {
  return x.c < y.c || x.b < y.b || x.a < y.a;
}

__device__ static U3 hasFactor2(unsigned p, U3 m) {
  unsigned mp0 = mprime0(m);
  int mShift = __clz(m.c) - 2;
  assert(mShift >= 0);
  U3 flushedM = shl(m, mShift);
  unsigned R = 0xffffffffffffffffULL / ((0x100000000ULL | shl(flushedM.b, flushedM.c, 3)) + 1);
  U3 a = mod((U4){0, 0, 0, 1}, flushedM, mShift, R);
  print(a);
  for (int i = 0; i < 32; ++i) {
    U6 a2 = square(a);
    a = montRed(a2, m, mp0);
    if (p & 0x80000000) { a = shl(a, 1); }
    // printf("%2d: ", i); print(a);
    p <<= 1;
  }
  return montRed(makeU6(a), m, mp0);
}

// Return whether m is a factor of 2^power - 1
// power < 2^30. m >= 2**64 && m < 2^94.
__device__ static bool hasFactor(unsigned p, U3 m) {
  assert(!(p >> 30));
  // Will consume p in 6 slices of 5bits each. The active slice of 5bits is flushed to the top of p.
  p <<= 2;
  
  assert(m.c);
  int mShift = __clz(m.c) - 2;
  assert(mShift >= 0);
  U3 flushedM = shl(m, mShift);
  unsigned R = 0xffffffffffffffffULL / ((0x100000000ULL | shl(flushedM.b, flushedM.c, 3)) + 1);

  printf("p %d shift %d\n", (p >> 27), mShift);
  U3 a = mod((U4) {0, 0, 0, 1 << (p >> 27)}, flushedM, mShift, R);
  print(a);
  unsigned mp0 = mprime0(m);
  print(montRed(makeU6(a), m, mp0));
  
  p <<= 5;
  for (int i = 0; i < 4; ++i, p <<= 5) {
    printf("starting slice %d\n", i);
    a = power32(a, m, mp0);
    a = mod(shl(makeU4(a), p >> 27), flushedM, mShift, R);
    print(a);
    // print(montRed(makeU6(a), m, mp0));
  }
  
  a = power32(a, m, mp0);
  a = montRed(makeU6(shl(makeU4(a), p >> 27)), m, mp0);
  assert(less(a, m));
  return a.a == 1 && !a.b && !a.c;
}

__managed__ U3 out;

__global__ void test(unsigned p, U3 m) {
  out = hasFactor2(p, m);
}

struct Test {
  unsigned p;
  u64 k;
};

Test tests[] = {
  {800000479,      3089082494924ull},
  {800000011,       955047115584ull},
  {500953,    895189023449202923ull},
  {119129573,     12186227586967ull},
};

int main() {
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  for (Test t : tests) {
    unsigned p = t.p;
    u64 k = t.k;
    u128 mm = (2 * p) * (u128) k + 1;
    U3 m = makeU3(mm);
    // printf("%uL\n", k);
    print(m);
    test<<<1, 1>>>(p, m);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
      break;
    } else {
      print(out);
    }
  }
}

  // U3 m{0xf817da27, 0x65e1be70, 0x0000009d};

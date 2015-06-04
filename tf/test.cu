// Copyright (c) Mihai Preda, 2015.

#include "common.h"
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>

#define THREADS_PER_BLOCK 512
#define NCLASS     (4 * 3 * 5 * 7 * 11)
#define NGOODCLASS (2 * 2 * 4 * 6 * 10)

#define SIEVE_THREADS 32
#define WORK_THREADS (THREADS_PER_BLOCK - SIEVE_THREADS)
#define NWORDS (12 * 1024)
#define NBITS (NWORDS << 5)
#define NPRIMES (ASIZE(primes))

__device__ const u32 primes[] = {
#include "primes-512k.inc"
};

__managed__ u64 foundFactor;
__managed__ u16 classTab[NGOODCLASS];
__device__ u32 classBtcTab[NGOODCLASS][NPRIMES];

struct U2 { unsigned a, b; };
struct U3 { unsigned a, b, c; };
struct U4 { unsigned a, b, c, d; };
struct U5 { unsigned a, b, c, d, e; };
struct U6 { unsigned a, b, c, d, e, f; };

u64 timeMillis() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

#define INLINE extern inline __device__

INLINE __host__ void print(U3 a) {
  printf("0x%08x%08x%08x\n", a.c, a.b, a.a);
}

INLINE __host__ void print(U4 a) {
  printf("0x%08x%08x%08x%08x\n", a.d, a.c, a.b, a.a);
}

INLINE __host__ void print(U5 a) {
  printf("0x%08x%08x%08x%08x%08x\n", a.e, a.d, a.c, a.b, a.a);
}

INLINE __host__ void print(U6 a) {
  printf("0x%08x%08x%08x%08x%08x%08x\n", a.f, a.e, a.d, a.c, a.b, a.a);
}

#ifdef NDEBUG
#define print(x)
#endif

// Funnel shift left.
INLINE unsigned shl(unsigned a, unsigned b, int n) {
  unsigned r;
  asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(n));
  return r;
}

// Funnel shift right.
INLINE unsigned shr(unsigned a, unsigned b, int n) {
  unsigned r;
  asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(n));
  return r;
}

__device__ static U6 add(U6 x, U6 y) {
  unsigned a, b, c, d, e, f;
  asm("add.cc.u32  %0, %6,  %12;"
      "addc.cc.u32 %1, %7,  %13;"
      "addc.cc.u32 %2, %8,  %14;"
      "addc.cc.u32 %3, %9,  %15;"
      "addc.cc.u32 %4, %10, %16;"
      "addc.u32    %5, %11, %17;"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d), "=r"(e), "=r"(f)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(x.d), "r"(x.e), "r"(x.f),
        "r"(y.a), "r"(y.b), "r"(y.c), "r"(y.d), "r"(y.e), "r"(y.f));
  return (U6) {a, b, c, d, e, f};
}

__device__ U5 add(U5 x, U3 y) {
 unsigned a, b, c, d, e;
  asm("add.cc.u32  %0, %5, %10;"
      "addc.cc.u32 %1, %6, %11;"
      "addc.cc.u32 %2, %7, %12;"
      "addc.cc.u32 %3, %8, 0;"
      "addc.u32    %4, %9, 0;"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d), "=r"(e)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(x.d), "r"(x.e),
        "r"(y.a), "r"(y.b), "r"(y.c));
  return (U5) {a, b, c, d, e};
}

__device__ U4 add(U4 x, U3 y) {
 unsigned a, b, c, d;
  asm("add.cc.u32  %0, %4, %8;"
      "addc.cc.u32 %1, %5, %9;"
      "addc.cc.u32 %2, %6, %10;"
      "addc.u32    %3, %7, 0;"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(x.d), "r"(y.a), "r"(y.b), "r"(y.c));
  return (U4) {a, b, c, d};
}

__device__ static U3 add(U3 x, U3 y) {
  unsigned a, b, c;
  asm("add.cc.u32  %0, %3, %6;"
      "addc.cc.u32 %1, %4, %7;"
      "addc.u32    %2, %5, %8;"
      : "=r"(a), "=r"(b), "=r"(c)
      : "r"(x.a), "r"(x.b), "r"(x.c),
        "r"(y.a), "r"(y.b), "r"(y.c));
  return (U3) {a, b, c};
}

__device__ static U3 sub(U3 x, U3 y) {
  unsigned a, b, c;
  asm("sub.cc.u32  %0, %3, %6;"
      "subc.cc.u32 %1, %4, %7;"
      "subc.u32    %2, %5, %8;"
      : "=r"(a), "=r"(b), "=r"(c)
      : "r"(x.a), "r"(x.b), "r"(x.c),
        "r"(y.a), "r"(y.b), "r"(y.c));
  return (U3) {a, b, c};
}

__device__ static U4 sub(U4 x, U4 y) {
  unsigned a, b, c, d;
  asm("sub.cc.u32  %0, %4, %8;"
      "subc.cc.u32 %1, %5, %9;"
      "subc.cc.u32 %2, %6, %10;"
      "subc.u32    %3, %7, %11;"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(x.d),
        "r"(y.a), "r"(y.b), "r"(y.c), "r"(y.d));
  return (U4) {a, b, c, d};
}

// return x - (y << 32)
__device__ U5 subShl1w(U5 x, U4 y) {
  U4 t = sub((U4) {x.b, x.c, x.d, x.e}, y);
  return (U5) {x.a, t.a, t.b, t.c, t.d};
}

__device__ static U4 mul(U3 x, unsigned n) {
  unsigned a, b, c, d;
  asm(
      "mul.hi.u32     %1, %4, %7;"
      "mul.lo.u32     %2, %6, %7;"
      "mad.lo.cc.u32  %1, %5, %7, %1;"
      "mul.lo.u32     %0, %4, %7;"
      "madc.hi.cc.u32 %2, %5, %7, %2;"
      "madc.hi.u32    %3, %6, %7, 0;"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(n));
  return (U4) {a, b, c, d};
}

// return (x*n >> 32) + 1
__device__ U3 mulM(U3 x, u32 n) {
  unsigned a, b, c;
  asm(
      "set.ne.u32.u32 %0,  0, %6;"
      "neg.s32        %0, %0;"
      "mad.hi.u32     %0, %3, %6, %0;"
      "mul.lo.u32     %1, %5, %6;"
      "mad.lo.cc.u32  %0, %4, %6, %0;"
      "madc.hi.cc.u32 %1, %4, %6, %1;"
      "madc.hi.u32    %2, %5, %6, 0;"
      : "=r"(a), "=r"(b), "=r"(c)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(n));
  return (U3) {a, b, c};
}

__device__ static U3 mul(U2 x, unsigned n) {
  unsigned a, b, c;
  asm(
      "mul.hi.u32     %1, %3, %5;"
      "mul.lo.u32     %0, %3, %5;"
      "mad.lo.cc.u32  %1, %4, %5, %1;"
      "madc.hi.u32    %2, %4, %5, 0;"
      : "=r"(a), "=r"(b), "=r"(c)
      : "r"(x.a), "r"(x.b), "r"(n));
  return (U3) {a, b, c};
}

__device__ static U4 square(U2 x) {
  unsigned a, b, c, d;
  asm(
      "mul.lo.u32     %1, %4, %5;"
      "mul.hi.u32     %2, %4, %5;"
      "mul.lo.u32     %0, %4, %4;"
      "add.cc.u32     %1, %1, %1;"
      "addc.cc.u32    %2, %2, %2;"
      "addc.u32       %3, 0, 0;"
      
      "mad.hi.cc.u32  %1, %4, %4, %1;"
      "madc.lo.cc.u32 %2, %5, %5, %2;"
      "madc.hi.u32    %3, %5, %5, %3;"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
      : "r"(x.a), "r"(x.b));
  return (U4) {a, b, c, d};
}

__device__ static U6 square(U3 x) {
  U2 ab = {x.a, x.b};
  U4 ab2 = square(ab);
  U3 abc = mul(ab, x.c + x.c);
  
  unsigned c, d, e, f;
  asm(
      "add.cc.u32  %0, %4, %6;"
      "addc.cc.u32 %1, %5, %7;"
      "mul.hi.u32  %3, %9, %9;"
      "madc.lo.cc.u32 %2, %9, %9, %8;"
      "addc.u32       %3, %3, 0;"
      : "=r"(c), "=r"(d), "=r"(e), "=r"(f)
      : "r"(ab2.c), "r"(ab2.d), "r"(abc.a), "r"(abc.b), "r"(abc.c), "r"(x.c));
  assert(!(f & 0xc0000000));
  return (U6) {ab2.a, ab2.b, c, d, e, f};
}

// square 80bit -> 160bit
__device__ static U5 square80(U3 x) {
  assert(!(x.c & 0xffff0000));
  U2 ab = {x.a, x.b};
  U4 ab2 = square(ab);
  U3 abc = mul(ab, x.c + x.c);
  
  unsigned c, d, e;
  asm(
      "add.cc.u32  %0, %3, %5;"
      "addc.cc.u32 %1, %4, %6;"
      "madc.lo.u32 %2, %8, %8, %7;"
      : "=r"(c), "=r"(d), "=r"(e)
      : "r"(ab2.c), "r"(ab2.d), "r"(abc.a), "r"(abc.b), "r"(abc.c), "r"(x.c));
  return (U5) {ab2.a, ab2.b, c, d, e};
}


INLINE U5 shl1w(U4 x)  { return (U5) {0, x.a, x.b, x.c, x.d}; }
INLINE U6 shl2w(U4 x)  { return (U6) {0, 0, x.a, x.b, x.c, x.d}; }
INLINE U6 makeU6(U3 x) { return (U6) {x.a, x.b, x.c, 0, 0, 0}; }
INLINE U5 makeU5(U4 x) { return (U5) {x.a, x.b, x.c, x.d, 0}; }
INLINE U6 makeU6(U5 x) { return (U6) {x.a, x.b, x.c, x.d, x.e, 0}; }
INLINE U6 makeU6(U4 x) { return makeU6(makeU5(x)); }
INLINE U2 makeU2(u64 x) { return (U2) {(unsigned) x, (unsigned) (x >> 32)}; }
__device__ U3 negative(U3 x) {
  return (U3) {-x.a, ~x.b + (!x.a), ~x.c + (!x.a && !x.b)};
}
__device__ bool ge(U3 x, U3 y) {
  return (x.c > y.c)
    || (x.c == y.c && x.b > y.b)
    || (x.c == y.c && x.b == y.b && x.a >= y.a);
}

__device__ static U3 shl(U3 x, int n) {
  assert(n >= 0 && n < 32 && !(x.c >> (32 - n)));
  return (U3) {x.a << n, shl(x.a, x.b, n), shl(x.b, x.c, n)};
}

/*
__device__ static U3 shr(U3 x, int n) {
  assert(n >= 0 && n < 32);
  return (U3) {shr(x.a, x.b, n), shr(x.b, x.c, n), x.c >> n};
}
*/

__device__ static U4 shl(U4 x, int n) {
  // assert(n >= 0 && n < 32 && !(x.d >> (32 - n)));
  return (U4) {x.a << n, shl(x.a, x.b, n), shl(x.b, x.c, n), shl(x.c, x.d, n)};
}

__device__ U5 shl(U5 x, int n) {
  assert(n >= 0 && n < 32 && !(x.e >> (32 - n)));
  U4 t = shl((U4) {x.a, x.b, x.c, x.d}, n);
  return (U5) {t.a, t.b, t.c, t.d, shl(x.d, x.e, n)};
}

__device__ U3 mod(U4 x, U3 m) {
  assert(m.c);
  assert(!(m.c & 0x80000000));
  int shift = __clz(m.c) - 2;
  assert(shift >= 0);
  m = shl(m, shift);
  unsigned R = 0xffffffffffffffffULL / ((0x100000000ULL | shl(m.b, m.c, 3)) + 1);

  unsigned n = mulhi(x.d, R);
  x = sub(x, shl(mul(m, n), 3));
  assert(!(x.d & 0xfffffffc));
  n = mulhi(shl(x.c, x.d, 28), R) >> 25;
  x = sub(x, mul(m, n));
  assert(!x.d);
  return (U3) {x.a, x.b, x.c};
}

__device__ U3 modx(U4 x, U3 m) {
  assert(m.c);
  assert(!(m.c & 0x80000000));
  int shift = __clz(m.c) - 2;
  assert(shift >= 0);
  m = shl(m, shift);
  unsigned R = 0xffffffffffffffffULL / ((0x100000000ULL | shl(m.b, m.c, 3)) + 1);
  // unsigned R = 0x7fffffff00000001 / shl(m.b, m.c, 2);


  unsigned n = mulhi(x.d, R);
  x = sub(x, shl(mul(m, n), 3));
  assert(!(x.d & 0xfffffffc));
  n = mulhi(shl(x.c, x.d, 28), R) >> 25;
  x = sub(x, mul(m, n));
  assert(!x.d);
  return (U3) {x.a, x.b, x.c};
}

__device__ U3 mod(U5 x, U3 m) {
  assert(m.c);
  int shift = __clz(m.c) + 1;
  assert(shift >= 18);
  unsigned R = 0xffffffffffffffffULL / ((0x100000000ULL | shl(m.b, m.c, shift)) + 1);
  unsigned n;
  
  n = mulhi(x.e, R);
  x = subShl1w(x, shl(mul(m, n), shift));
  assert(!(x.e & 0xfffffff0));
  x = shl(x, 28);
  
  n = mulhi(x.e, R);
  x = subShl1w(x, shl(mul(m, n), shift));
  assert(!(x.e & 0xfffffff0));
  x = shl(x, 28);
  
  n = mulhi(x.e, R) & 0xffffffc0;
  x = subShl1w(x, shl(mul(m, n), shift));
  assert((!(x.e & 0xffffff80)));
  x = shl(x, 8);
  assert(!(x.e & 0xffff8000));
  return (U3) {x.c, x.d, x.e};
}

// Compute m' such that: (unsigned) (m * m') == 0xffffffff, using extended binary euclidian algorithm.
// See http://www.ucl.ac.uk/~ucahcjm/combopt/ext_gcd_python_programs.pdf
// m must be odd.
__device__ static unsigned mprime(unsigned m) {
  m = (m >> 1) + 1;
  unsigned u = m;
  unsigned v = m << 31; 
  for (int i = 0; i < 30; ++i) {
    u = (u >> 1) + ((u & 1) ? m : 0);
    v = shr(v, u, 1);
  }
  return v | 1;
}

// Montgomery Reduction
// See https://www.cosic.esat.kuleuven.be/publications/article-144.pdf
// Returns x * U^-1 mod m
__device__ static U3 montRed1(U6 x, U3 m, unsigned mp0) {
  assert(!(x.f & 0xc0000000));
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
  assert(!(x.f & 0xc0000000));
  return (U3) {x.d, x.e, x.f};
}

__device__ static U3 montRed(U6 x6, U3 m, unsigned mp) {
  assert(!(x6.f & 0xc0000000));
  U5 x5 = (U5) {x6.b, x6.c, x6.d, x6.e, x6.f};
  x5 = add(x5, mulM(m, x6.a * mp));
  U4 x4 = (U4) {x5.b, x5.c, x5.d, x5.e};
  x4 = add(x4, mulM(m, x5.a * mp));
  U3 x3 = (U3) {x4.b, x4.c, x4.d};
  x3 = add(x3, mulM(m, x4.a * mp));
  assert(!(x3.c & 0xc0000000));
  return x3;
}

// returns 2^exp % m
__device__ U3 expMod1(u32 exp, U3 m) {
  assert(exp & 0x80000000);
  unsigned mp0 = mprime(m.a);

  int sh = exp >> 27;
  assert(sh >= 16 && sh < 32);
  U3 a = modx((U4){0, 0, 0, 1 << sh}, m);
  for (exp <<= 5; exp; exp += exp) {
    a = montRed(square(a), m, mp0);
    if (exp & 0x80000000) { a = shl(a, 1); }
  }
  return montRed(makeU6(a), m, mp0);
}

// returns 2^exp % m
__device__ U3 expMod2(u32 exp, U3 m) {
  assert(exp & 0x80000000);
  unsigned mp0 = mprime(m.a);

  int sh = exp >> 26;
  assert(sh >= 32 && sh < 64);
  U3 a = mod((U5){0, 0, 0, 0, 1 << (sh - 32)}, m);
  for (exp <<= 6; exp; exp += exp) {
    U6 a2 = square(a);
    a = montRed(a2, m, mp0);
#ifndef NDEBUG
    U3 b = montRed1(a2, m, mp0);
    assert(a.a == b.a && a.b == b.b && a.c == b.c);
#endif
    if (exp & 0x80000000) { a = shl(a, 1); }
  }
  return montRed(makeU6(a), m, mp0);
}

__device__ U3 expMod3(u32 exp, U3 m) {
  assert(exp & 0x80000000);
  int sh = exp >> 25;
  assert(sh >= 64 && sh < 128);
  U3 a = mod((U4){0, 0, 1 << (sh - 64), 1 << (sh - 96)}, m);
  for (exp <<= 7; exp; exp += exp) {
    a = mod(square80(a), m);
    if (exp & 0x80000000) { a = shl(a, 1); }
  }
  return a; // TODO: final mod.
}

// return 2 * k * p + 1 as U3
__device__ U3 makeQ(unsigned p, u64 k) {
  return add(mul(makeU2(k), p + p), (U3){1, 0, 0});
}

// returns whether (2*k*p + 1) is a factor of (2^p - 1)
__device__ bool isFactor(u32 exp, u32 flushedExp, u64 k) {
  U3 q = makeQ(exp, k);
  U3 r = expMod2(flushedExp, q);
#ifndef NDEBUG
  U3 r2 = expMod1(flushedExp, q);
  if (!(r.a == r2.a && r.b == r2.b && r.c == r2.c)) {
    printf("m 0x%08x%08x%08x r1 0x%08x%08x%08x r2 0x%08x%08x%08x\n",
           q.c, q.b, q.a, r.c, r.b, r.a, r2.c, r2.b, r2.a);
  }
  assert(r.a == r2.a && r.b == r2.b && r.c == r2.c);
#endif
  return r.a == 1 && !r.b && !r.c;
}

__device__ u32 modInv32(u64 step, u32 prime) {
  int n = step % prime;
  int q = prime / n;
  int d = prime - q * n;
  int x = -q;
  int prevX = 1;
  while (d) {
    q = n / d;
    { int save = d; d = n - q * d; n = save; }            // n = set(d, n - q * d);
    { int save = x; x = prevX - q * x; prevX = save; }    // prevX = set(x, prevX - q * x);
  }
  return (prevX >= 0) ? prevX : (prevX + prime);
}

#define ID (blockIdx.x * blockDim.x + threadIdx.x)

// 3 times 64bit modulo, very expensive!
__device__ int bitToClear(u32 exp, u64 k, u32 prime, u32 inv) {
  u32 kmod = k % prime;
  u32 qmod = (kmod * (u64) (exp << 1) + 1) % prime;
  return (prime - qmod) * (u64) inv % prime;
}

__device__ int bfind(u32 x) { int r; asm("bfind.u32 %0, %1;": "=r"(r): "r"(x)); return r; }

__global__ void __launch_bounds__(1024) initBtcTab(u32 exp, u64 k0, u64 step) {
  for (const u32 *p = primes + ID, *end = primes + ASIZE(primes); p < end; p += gridDim.x * blockDim.x) {
    u32 prime = *p;
    u32 inv = modInv32(step, prime);
    for (int i = 0; i < NGOODCLASS; ++i) {
      classBtcTab[i][p - primes] = bitToClear(exp, k0 + classTab[i], prime, inv);
    }
  }
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) tf(u32 exp, u64 k0, u64 kEnd, int c0) {
  __shared__ u32 words[NWORDS];
#define LOCAL_WORDS ((NWORDS + WORK_THREADS - 1) / WORK_THREADS)
  u32 localWords[LOCAL_WORDS];
  localWords[0] = 0;
  u32 *localEnd = localWords;
  
  const int tid = threadIdx.x;
  const int cid = c0 + blockIdx.x;
  const int c = classTab[cid];
  const u32 flushedExp = exp << __clz(exp);
  u32 * const btcTab = classBtcTab[cid];
  u64 kBlock = k0 + c + (tid - SIEVE_THREADS) * (32 * NCLASS) - NWORDS * 32 * NCLASS;
  bool shouldExit = false;
  int round = 0;
  // u32 timeSieve, timeTest, timeCopy;
  
  while (kBlock < kEnd && !shouldExit && round < 100) {
    // u32 time0 = clock();
    if (tid < SIEVE_THREADS) {
      u32 *btcp = btcTab + tid;
      for (const u32 *p = primes + tid, *end = primes + ASIZE(primes); p < end; p += SIEVE_THREADS, btcp += SIEVE_THREADS) {
        int prime = *p;
        int btc = *btcp;
        while (btc < NBITS) {
          atomicOr(words + (btc >> 5), 1 << (btc & 0x1f));
          btc += prime;
        }
        *btcp = btc - NBITS;
      }
      // timeSieve = clock() - time0;
    } else {
      u32 *p = localWords;
      u32 bits = *p;
      u64 kWord = kBlock;
      while (true) {
        while (!bits) {
          kWord += WORK_THREADS * (32 * NCLASS);
          if (++p >= localEnd || kWord >= kEnd) { goto out; }
          bits = *p;
        }
        int bit = bfind(bits);
        bits &= ~(1 << bit);
        if (isFactor(exp, flushedExp, kWord + bit * NCLASS)) {
          foundFactor = kWord + bit * NCLASS;
        }
      }
    }
  out:
    __syncthreads();
    /*
    u32 tmp = clock();
    timeTest = tmp - time0;
    time0 = tmp;
    */
    
    shouldExit = foundFactor;
    int pops = 0;
    if (tid >= SIEVE_THREADS) {
      u32 *out = localWords;
      for (u32 *p = words + (tid - SIEVE_THREADS), *end = words + NWORDS; p < end; p += WORK_THREADS, ++out) {
        u32 tmp = ~*p;
        *p = 0;
        *out = tmp;
        pops += __popc(tmp);
      }
      localEnd = out;
    }
    __syncthreads();
    /*
    timeCopy = clock() - time0;
    if (!tid) {
      printf("class %d round %d sieve %u\n", cid, round, (timeSieve >> 10));
    } else if (tid == SIEVE_THREADS) {
      printf("class %d round %d test %u copy %u pops %d\n", cid, round, (timeTest >> 10), (timeCopy >> 10), pops);
    }
    */
    kBlock += NWORDS * 32 * NCLASS;
    ++round;
  }
}

void initClasses(u32 exp) {
  int nClass = 0;
  for (int c = 0; c < NCLASS; ++c) {
    if (acceptClass(exp, c)) {
      classTab[nClass++] = (u16) c;
    }
  }
  assert(nClass == NGOODCLASS);
}

u64 calculateK(u32 exp, int bits) {
  u64 k = (((u128) 1) << (bits - 1)) / exp;
  return k - k % NCLASS;
}

#define CUDA_CHECK_ERR  err = cudaGetLastError(); if (err) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return; }

int main() {
  cudaError_t err;
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  // cudaSetDevice(1);
  CUDA_CHECK_ERR;
  
  const u32 exp = 119904229;
  const u64 step = 2 * NCLASS * (u64) exp;
  int startPow2 = 67;
  const u64 k0   = calculateK(exp, startPow2);
  const u64 kEnd = calculateK(exp, startPow2 + 1);
  initClasses(exp);
  printf("exp %u k0 %llu kEnd %llu\n", exp, k0, kEnd);
  
#define BTC_THREADS 1024
#define BTC_BLOCKS (NPRIMES / BTC_THREADS)
  assert(NPRIMES % BTC_THREADS == 0);
  u64 t1 = timeMillis();
  initBtcTab<<<BTC_BLOCKS, BTC_THREADS>>>(exp, k0, step);
  cudaDeviceSynchronize();
  u64 t2 = timeMillis();
  printf("initBtcTab: blocks %lu time %llu\n", BTC_BLOCKS, (t2 - t1));
  CUDA_CHECK_ERR;
  //return;
#define BLOCKS 64
  assert(NGOODCLASS % BLOCKS == 0);
  u64 t3 = timeMillis();
  for (int cid = 0; cid < NGOODCLASS; cid += BLOCKS) {
    tf<<<BLOCKS, THREADS_PER_BLOCK>>>(exp, k0, kEnd, cid);
    cudaDeviceSynchronize();
    u64 t4 = timeMillis();
    printf("class %d time %llu\n", cid, (t4 - t3));
    t3 = t4;
  }
  printf("TF: %llu\n", (t3 - t2));
  CUDA_CHECK_ERR;
  if (foundFactor) {
    printf("Found factor %llu\n", foundFactor);
  }
  // cudaDeviceReset();
}

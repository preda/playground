// Copyright (c) Mihai Preda, 2015.

// #include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>

#define THREADS_PER_BLOCK 1024
#define BLOCKS_PER_GRID 64

// #define assert(x) 

typedef unsigned long long u64;
typedef unsigned short u16;
typedef __uint128_t u128;

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

#define print(x)

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

__device__ static U3 mul(U2 x, unsigned n) {
  unsigned a, b, c;
  asm("mul.lo.u32     %0, %3, %5;"
      "mul.hi.u32     %1, %3, %5;"
      
      "mad.lo.cc.u32  %1, %4, %5, %1;"
      "madc.hi.u32    %2, %4, %5, 0;"
      : "=r"(a), "=r"(b), "=r"(c)
      : "r"(x.a), "r"(x.b), "r"(n));
  return (U3) {a, b, c};
}

__device__ static U4 mul(U3 x, unsigned n) {
  unsigned a, b, c, d;
  asm("mul.lo.u32     %0, %4, %7;"
      "mul.hi.u32     %1, %4, %7;"
      
      "mad.lo.cc.u32  %1, %5, %7, %1;"
      "madc.hi.u32    %2, %5, %7, 0;"
      
      "mad.lo.cc.u32  %2, %6, %7, %2;"
      "madc.hi.u32    %3, %6, %7, 0;"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(n));
  return (U4) {a, b, c, d};
}

__device__ static U4 square(U2 x) {
  unsigned a, b, c, d;
  asm("mul.lo.u32     %0, %4, %4;"
      "mul.lo.u32     %1, %4, %5;"
      "mul.hi.u32     %2, %4, %5;"
      
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
  assert(!(x.c & 0x80000000));
  U2 ab = {x.a, x.b};
  U4 ab2 = square(ab);
  U3 abc = mul(ab, x.c + x.c);
  
  unsigned c, d, e, f;
  asm("add.cc.u32     %0, %4, %6;"
      "addc.cc.u32    %1, %5, %7;"
      "madc.lo.cc.u32 %2, %9, %9, %8;"
      "madc.hi.u32    %3, %9, %9, 0;"
      : "=r"(c), "=r"(d), "=r"(e), "=r"(f)
      : "r"(ab2.c), "r"(ab2.d), "r"(abc.a), "r"(abc.b), "r"(abc.c), "r"(x.c));
  return (U6) {ab2.a, ab2.b, c, d, e, f};
}

INLINE U5 shl1w(U4 x)  { return (U5) {0, x.a, x.b, x.c, x.d}; }
INLINE U6 shl2w(U4 x)  { return (U6) {0, 0, x.a, x.b, x.c, x.d}; }
INLINE U6 makeU6(U3 x) { return (U6) {x.a, x.b, x.c, 0, 0, 0}; }
INLINE U6 makeU6(U4 x) { return (U6) {x.a, x.b, x.c, x.d, 0, 0}; }
INLINE U6 makeU6(U5 x) { return (U6) {x.a, x.b, x.c, x.d, x.e, 0}; }
INLINE U2 makeU2(u64 x) { return (U2) {(unsigned) x, (unsigned) (x >> 32)}; }

__device__ static U3 shl(U3 x, int n) {
  assert(n >= 0 && n < 32 && !(x.c >> (32 - n)));
  return (U3) {x.a << n, shl(x.a, x.b, n), shl(x.b, x.c, n)};
}

__device__ static U4 shl(U4 x, int n) {
  assert(n >= 0 && n < 32 && !(x.d >> (32 - n)));
  return (U4) {x.a << n, shl(x.a, x.b, n), shl(x.b, x.c, n), shl(x.c, x.d, n)};
}

// Relaxed modulo: result < 2^96. m >= 2^64 && m < 2^94.
__device__ U3 mod(U4 x, U3 m) {
  assert(m.c);
  int shift = __clz(m.c) - 2;
  assert(shift >= 0);
  m = shl(m, shift);
  unsigned R = 0xffffffffffffffffULL / ((0x100000000ULL | shl(m.b, m.c, 3)) + 1);

  unsigned n = mulhi(x.d, R);
  x = sub(x, shl(mul(m, n), 3));
  assert(!(x.d & 0xfffffff0));
  n = mulhi(shl(x.c, x.d, 28), R) >> 25;
  x = sub(x, mul(m, n));
  assert(!x.d);
  return (U3) {x.a, x.b, x.c};
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
// Returns x * U^-1 mod m
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
  return (U3) {x.d, x.e, x.f};
}

// returns 2^p % m
__device__ static U3 expMod(unsigned p, U3 m) {
  unsigned mp0 = mprime0(m);

  U3 a = mod((U4){0, 0, 0, (1 << (p >> 27))}, m);
  for (p <<= 5; p; p += p) {
    // print(a);
    a = montRed(square(a), m, mp0);
    if (p & 0x80000000) { a = shl(a, 1); }
  }
  return montRed(makeU6(a), m, mp0);
}

// return 2 * k * p + 1 as U3
__device__ static U3 makeQ(unsigned p, u64 k) {
  return add(mul(makeU2(k), p + p), (U3){1, 0, 0});
}

// returns whether (2*k*p + 1) is a factor of (2^p - 1)
__device__ static bool isFactor(unsigned p, u64 k) {
  U3 q = makeQ(p, k);
  U3 r = expMod(p, q);
  return r.a == 1 && !r.b && !r.c;
}

#define NTHREADS (THREADS_PER_BLOCK * BLOCKS_PER_GRID)

__managed__ u64 deviceFactor;
__managed__ unsigned deviceClasses[NTHREADS];
__managed__ U3 out;

// #define NCLASS (4 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23)
#define NCLASS (4 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23)

__global__ void tf(unsigned p, u64 k0, unsigned *classes, int repeat) {
  unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned c = deviceClasses[id];
  if (c == 0xffffffff) { return; }
  u64 k = k0 + c;
  for (int i = repeat; i > 0; --i) {
    if (isFactor(p, k)) {
      printf("%d found factor %llu\n", id, k);
      deviceFactor = k;
      break;
    }
    k += NCLASS;
  }
}

bool launch(unsigned p, u64 k0, int t, unsigned *classes, int repeat) {
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return true;
  }
  if (deviceFactor) {
    printf("factor %llu\n", deviceFactor);
    return true;
  }
  memcpy(deviceClasses, classes, t * sizeof(unsigned));
  if (t < NTHREADS) {
    printf("Tail %d\n", t);
    memset(deviceClasses + t, 0xff, (NTHREADS - t) * sizeof(unsigned));
  }
  tf<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(p, k0, deviceClasses, repeat);
  return false;
}

// whether 2 * k * p + 1 == 1 or 7 modulo 8.
extern inline bool q1or7mod8(unsigned p, u64 k) {
  return !(k & 3) || ((k & 3) + (p & 3) == 4);
}

// whether 2 * k * p + 1 != 0 modulo prime
extern inline bool notMultiple(unsigned p, unsigned k, unsigned prime) {
  // return (((k + k) % prime) * (p % prime) + 1) % prime != 0;
  unsigned kk = k % prime;
  return !kk || ((p % prime) * kk * 2 + 1) % prime != 0;
  
  // return ((p % prime) * 2 * (u64)k + 1) % prime != 0;
}

static bool accept(unsigned p, unsigned k) {
  return q1or7mod8(p, k) && notMultiple(p, k, 3) && notMultiple(p, k, 5) && notMultiple(p, k, 7)
    && notMultiple(p, k, 11) && notMultiple(p, k, 13) && notMultiple(p, k, 17)
    && notMultiple(p, k, 19) && notMultiple(p, k, 23);
}

int findFactor(unsigned p, u64 k0, int repeat) {
  u64 timeStart = timeMillis();
  u64 time1 = timeStart;
  unsigned classes[NTHREADS];
  int accepted = 0;
  int t = 0;
  int c = 0;
  int nLaunch = 0;
  for (; c <= NCLASS - 4; c += 4) {
    if (accept(p, c))     { classes[t++] = c; }
    if (accept(p, c + 1)) { classes[t++] = c; }
    if (accept(p, c + 2)) { classes[t++] = c; }
    if (accept(p, c + 3)) { classes[t++] = c; }
    
    if (t >= NTHREADS) {
      accepted += NTHREADS;
      t = 0;
      ++nLaunch;
      if (launch(p, k0, NTHREADS, classes, repeat)) { return -1; }
      if (!(nLaunch & 0xf)) {
        u64 time2 = timeMillis();
        printf("%8u: %u ms\n", c, (unsigned)(time2 - time1));
        time1 = time2;
      }
    }
  }
  for (; c < NCLASS; c++) {
    if (accept(p, c)) { classes[t++] = c; }
  }
  accepted += t;
  launch(p, k0, t, classes, repeat);
  u64 time2 = timeMillis(); time1 = time2;
  printf("%8u: %u ms; total %llu\n", c, (unsigned)(time2 - time1), time2 - timeStart);
  return accepted;
}

int main() {
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  cudaError_t err = cudaGetLastError();
  printf("%s\n", cudaGetErrorString(err));
  
  const unsigned p = 119904229;
  int startPow2 = 67;
  u64 auxK = (((u128) 1) << (startPow2 - 1)) / p;
  u64 k0 = auxK - auxK % NCLASS;
  int repeat = (int) (auxK / NCLASS) + 1;
  printf("p %u K0 %llu threads %d classes %d repeat %d\n", p, k0, NTHREADS, NCLASS, repeat);
  int accepted = findFactor(p, k0, repeat);
  printf("accepted %d (%f%%)\n", accepted, accepted/(float)NCLASS*100);
}

struct Test { unsigned p; u64 k; };

#include "tests.inc"

__global__ void test(unsigned p, u64 k) {
  __shared__ unsigned shared[12 * 1024];
  shared[12 * 1024 - 1] = 1;
  U3 q = makeQ(p, k);
  out = expMod(p, q);
}

static void selfTest() {
  int n = sizeof(tests) / sizeof(tests[0]);
  for (Test *t = tests, *end = tests + n; t < end; ++t) {
    unsigned p = t->p;
    u64 k = t->k;
    int shift = __builtin_clz(p);
    assert(shift < 27);
    p <<= shift;
    // printf("p %u k %llu m: ", t->p, t->k); print(m);
    test<<<1, 1>>>(p, k);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(err));
      break;
    } else {
      if (out.a != 1 || out.b || out.c) {
        printf("ERROR %10u %20llu m ", t->p, t->k); print(m); print(out);
        break;
      } else {
        // printf("OK\n");
      }
    }
  }
}

#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

typedef unsigned long long u64;

struct U3 { unsigned a, b, c; };
struct U4 { unsigned a, b, c, d; };
struct U6 { unsigned a, b, c, d, e, f; };

__device__ static unsigned mul(unsigned a, unsigned b) { return a * b; }

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

__device__ static U4 mul(U3 v, unsigned n) {
  unsigned a =         mul(n, v.a);
  unsigned b =  add_cc(mulhi(n, v.a), mul(n, v.b));
  unsigned c = addc_cc(mulhi(n, v.b), mul(n, v.c));
  unsigned d =    addc(mulhi(n, v.c), 0);
  return {a, b, c, d};
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

__device__ static U6 shl2w(U4 x)  { return {0, 0, x.a, x.b, x.c, x.d}; }
__device__ static U6 shl1w(U4 x)  { return {0, x.a, x.b, x.c, x.d, 0}; }
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

// 3W = 6W % 3W; y >= 2**95
__device__ static U3 modAux(U6 x, U3 y) {
  assert(y.c & 0x80000000);
  const u64 R64 = 0xffffffffffffffffULL / ((0x100000000ULL | shl(y.b, y.c, 1)) + 1);
  assert((R64 >> 32) == 0);
  const unsigned R = (unsigned) R64;
  
  x = subshl(x, shl2w(mul(y, mulhi(x.f, R))), 1);
  assert((x.f & 0xfffffff0) == 0);
  x = subshl(x, shl1w(mul(y, mulhi(shl(x.e, x.f, 28), R))), 5);
  assert(x.f == 0 && (x.e & 0xffffff00) == 0);
  x = subshl(x, makeU6(mul(y, mulhi(shl(x.d, x.e, 24), R))), 9);
  assert(x.f == 0 && x.e == 0 && (x.d & 0xfffff000) == 0);
  x = subshl(x, makeU6(mul(y, mulhi(shl(x.c, x.d, 20), R) >> 19)), 0);
  assert(x.f == 0 && x.e == 0 && x.d == 0);
  
  return {x.a, x.b, x.c};
}

// y > 2**64
__device__ static U3 mod(U6 x, U3 y) {
  assert(y.c);
  int shift = __clz(y.c);
  assert(__clz(x.f) >= shift);
  return shr(modAux(shl(x, shift), shl(y, shift)), shift);
}

__device__ static U3 avg(U3 x, U3 y) {
  unsigned a =  add_cc(x.a, y.a);
  unsigned b = addc_cc(x.b, y.b);
  unsigned c = addc_cc(x.c, y.c);
  unsigned d = addc(0, 0);
  return {shr(a, b, 1), shr(b, c, 1), shr(c, d, 1)};
}

/*
__device__ static N96 add(N96 a, N96 b) {
  unsigned d0 = add_cc(a.d0, b.d0);
  unsigned d1 = addc_cc(a.d1, b.d1);
  unsigned d2 = addc(a.d2, b.d2);
  return {d0, d1, d2};
}
*/

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

static void print(U3 a) {
  printf("0x%08x%08x%08x\n", a.c, a.b, a.a);
}

__device__ static void printD(U3 a) {
  printf("0x%08x%08x%08x\n", a.c, a.b, a.a);
}

__global__ void test(U3 *out, U6 *as, U3 *bs) {
  /*
  N96 x = {2, 800, 16};
  N96 y = {4, 200, 1};
  printD(avg96(x, y));
  printD(shr(x, 1));
  printD(shr(y, 1));
  */
  U3 b = bs[0];
  gcd(b, out, out + 1);
}

#define N 32

__managed__ U6 as[N];
__managed__ U3 bs[N];
__managed__ U3 out[N * N];

int main() {
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  U3 b{0x12345fff, 0x224433aa, 0x76123229};
  bs[0] = b;
  test<<<1, 1>>>(out, as, bs);
  cudaDeviceSynchronize();

  print(b);
  print(out[0]);
  print(out[1]);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}

  /*
  u64 u, v;
  gcdHost(7, &u, &v);
  printf("%llx %llx\n", u, v);
  */

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

N96 random96() {
  N96 ret = {random32(), random32(), random32()};
  return ret;
}

N192 random192() {
  N192 ret = {random32(), random32(), random32(), random32(), random32(), random32()};
  return ret;
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

/*
__global__ void testMul2(unsigned *out, unsigned d0, unsigned d1, unsigned d2, unsigned n) {
  unsigned r;
  asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(d0), "r"(d1), "r"(n));
  *out = r;
  // int id = threadIdx.x;
  // mul96a(out, d0, d1, d2, n);
  // printf("%u: %x %x %x %x\n", n, r[0], r[1], r[2], r[3]);
}
*/

/*
// 6W = 6W - 6W
__device__ static void sub192(unsigned *out, unsigned *a, unsigned *b) {
  out[0] = sub_cc(a[0], b[0]);
  out[1] = subc_cc(a[1], b[1]);
  out[2] = subc_cc(a[2], b[2]);
  out[3] = subc_cc(a[3], b[3]);
  out[4] = subc_cc(a[4], b[4]);
  out[5] = subc(a[5], b[5]);
}
*/

/*
__device__ static void mul96a(unsigned *out, unsigned d0, unsigned d1, unsigned d2, unsigned n) {
  out[0] = mul(n, d0);
  out[1] = madhi_cc(n, d0, mul(n, d1));
  out[2] = madhic_cc(n, d1, mul(n, d2));
  out[3] = mulhic(n, d2);
}
*/

/*
__global__ void AAA(unsigned long long *out, unsigned long long a, unsigned long long b) {
  *out = a * b;
}

__global__ void BBB(unsigned *out, unsigned a, unsigned b, unsigned c) {
  unsigned r;
  asm("madc.lo.u32 %0, %1, %2, %3;": "=r"(r): "r"(a), "r"(b), "r"(c));
  *out = r;
}
*/


// #define mul(a, b) ((a) * (b))
  /*
  unsigned r;
  asm("mul.lo.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b));
  return r;
  */

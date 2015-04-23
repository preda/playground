#include <cuda.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

typedef unsigned long long u64;

typedef struct {
  unsigned d0, d1, d2;
} N96;

typedef struct {
  unsigned d0, d1, d2, d3, d4, d5;
} N192;

typedef struct {
  unsigned d0, d1, d2, d3;
} N128;

__device__ static unsigned mul(unsigned a, unsigned b) {
  return a * b;
}

__device__ static unsigned madhi_cc(unsigned a, unsigned b, unsigned c) {
  unsigned r;
  asm("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ static unsigned madhic_cc(unsigned a, unsigned b, unsigned c) {
  unsigned r;
  asm("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
  return r;
}

__device__ static unsigned mulhic(unsigned a, unsigned b) {
  unsigned r;
  asm("madc.hi.u32 %0, %1, %2, 0;" : "=r"(r) : "r"(a), "r"(b));
  return r;
}

__device__ static unsigned add_cc(unsigned a, unsigned b) {
  unsigned r;
  asm("add.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b));
  return r;
}

__device__ static unsigned addc_cc(unsigned a, unsigned b) {
  unsigned r;
  asm("addc.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b));
  return r;
}

__device__ static unsigned addc(unsigned a, unsigned b) {
  unsigned r;
  asm("addc.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b));
  return r;
}

__device__ static unsigned sub_cc(unsigned a, unsigned b) {
  unsigned r;
  asm("sub.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b));
  return r;
}

__device__ static unsigned subc_cc(unsigned a, unsigned b) {
  unsigned r;
  asm("subc.cc.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b));
  return r;
}

__device__ static unsigned subc(unsigned a, unsigned b) {
  unsigned r;
  asm("subc.u32 %0, %1, %2;" : "=r" (r) : "r" (a), "r" (b));
  return r;
}

__device__ static unsigned shfl(unsigned a, unsigned b, unsigned n) {
  unsigned r;
  asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(n));
  return r;
}

// 4W = 3W * 1W 
__device__ static N128 mul(N96 a, unsigned n) {
  N128 r;
  r.d0 =         mul(n, a.d0);
  r.d1 =  add_cc(mulhi(n, a.d0), mul(n, a.d1));
  r.d2 = addc_cc(mulhi(n, a.d1), mul(n, a.d2));
  r.d3 =    addc(mulhi(n, a.d2), 0);
  return r;
}

// 6W = 6W - 6W
__device__ static N192 subshl(N192 a, N192 b, int n) {
  N192 r;
  r.d0 =  sub_cc(a.d0, b.d0 << n);
  r.d1 = subc_cc(a.d1, shfl(b.d0, b.d1, n));
  r.d2 = subc_cc(a.d2, shfl(b.d1, b.d2, n));
  r.d3 = subc_cc(a.d3, shfl(b.d2, b.d3, n));
  r.d4 = subc_cc(a.d4, shfl(b.d3, b.d4, n));
  r.d5 =    subc(a.d5, shfl(b.d4, b.d5, n));
  return r;
}

__device__ static N192 shl64(N128 a) {
  N192 r = {0, 0, a.d0, a.d1, a.d2, a.d3};
  return r;
}

__device__ static N192 shl32(N128 a) {
  N192 r = {0, a.d0, a.d1, a.d2, a.d3, 0};
  return r;
}

__device__ static N192 shl0(N128 a) {
  N192 r = {a.d0, a.d1, a.d2, a.d3, 0, 0};
  return r;
}

// 3W = 6W % 3W; b >= 2**95
__device__ static N96 mod(N192 a, N96 b) {
  assert(b.d2 & 0x80000000);
  const u64 R64 = 0xffffffffffffffffULL / ((0x100000000ULL | shfl(b.d1, b.d2, 1)) + 1); // (((u64) b.d2) << 1) | (b.d1 >> 31));
  assert((R64 >> 32) == 0);
  const unsigned R = (unsigned) R64;
  unsigned n;
  N192 c;

  unsigned saveA5 = a.d5;
  
  n = mulhi(a.d5, R);
  c = shl64(mul(b, n));
  a = subshl(a, c, 1);
  if (a.d5 & 0xfffffff0) {
    printf("a5 %08x b2 %08x b1 %08x R %08x n %08x rem %08x\n", saveA5, b.d2, b.d1, R, n, a.d5);
  }
  assert((a.d5 & 0xfffffff0) == 0);

  n = mulhi(shfl(a.d4, a.d5, 28), R);
  c = shl32(mul(b, n));
  a = subshl(a, c, 5);
  assert(a.d5 == 0 && (a.d4 & 0xffffff00) == 0);

  n = mulhi(shfl(a.d3, a.d4, 24), R);
  c = shl0(mul(b, n));
  a = subshl(a, c, 9);
  assert(a.d5 == 0 && a.d4 == 0 && (a.d3 & 0xfffff000) == 0);

  n = mulhi(shfl(a.d2, a.d3, 20), R) >> 19;
  c = shl0(mul(b, n));
  a = subshl(a, c, 0);
  assert(a.d5 == 0 && a.d4 == 0 && a.d3 == 0);
  
  N96 ret = {a.d0, a.d1, a.d2};
  return ret;
}

__global__ void testMod(N96 *out, N192 *as, N96 *bs) {
  // printf("dim %d\n", blockDim.y);
  int x = threadIdx.x;
  int y = threadIdx.y;
  N192 a = as[x];
  N96 b = bs[y];
  out[x + y * blockDim.y] = mod(a, b);
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

#define N 32

__managed__ N192 as[N];
__managed__ N96 bs[N];
__managed__ N96 out[N * N];

int main() {
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
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
  testMod<<<1, blockDim>>>(out, as, bs);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }
}



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

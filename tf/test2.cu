// Copyright (c) Mihai Preda, 2015.

/*
  This is a program for trial factoring of Mersenne numbers, which are numbers of the form
  2^exp - 1. See http://www.mersenne.org/various/math.php

  For a given mersenne number 2^exp-1, i.e. for a given value "exp", exp being prime,
  the trial factors are of the form q = 2*k*exp + 1, and we're interested only in prime factors.

  Range: exp < 2^30. q < 2^94.

  A first step consist in generating prime candidate factors -- this is called "sieving" because
  it uses Erathostene's sieve. Next each factor q is tested by computing "modular exponentiation",
  reminder r = 2^exp modulo q. If this reminder is equal to 1, it means that q is a factor of
  2^exp-1, and thus the mersenne number is not prime.

  Both the sieving and the testing is run on the GPU.

  Some naming conventions used:
  shl: shift left.
  shr: shift right.
  U3: unsigned int using 3 words (i.e. 96bits).
  u32: unsigned int using 32 bits.
 */

#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/unistd.h>
#include "widemath.h"

#define ASIZE(a) (sizeof(a) / sizeof(a[0]))

// Threads per sieving block.
#define SIEV_THREADS 256
// Threads per testing block.
#define TEST_THREADS 512

// How many words of shared memory to use for sieving.
#define NWORDS (6 * 1024)

// Must update acceptClass() when changing these.
#define NCLASS     (4 * 3 * 5 * 7 * 11)
// Out of NCLASS, how many classes pass acceptClass(). Sync with NCLASS.
#define NGOODCLASS (2 * 2 * 4 * 6 * 10)

// Returns whether 2 * c * exp + 1 is 1 or 7 modulo 8.
// Any Marsenne factor must be of this form. See http://www.mersenne.org/various/math.php
bool q1or7mod8(u32 exp, u32 c) {
  return !(c & 3) || ((c & 3) + (exp & 3) == 4);
}

// whether 2 * c * exp + 1 != 0 modulo prime
bool notMultiple(u32 exp, u32 c, unsigned prime) { return (2 * c * (u64) exp + 1) % prime; }

bool acceptClass(u32 exp, u32 c) {
#define P(p) notMultiple(exp, c, p)
  return q1or7mod8(exp, c) && P(3) && P(5) && P(7) && P(11);
#undef P
}

// Bits for sieving.
#define NBITS (NWORDS << 5)

// Number of pre-computed primes for sieving.
#define NPRIMES (ASIZE(primes))

u64 timeMillis() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

DEVICE const u32 primes[] = {
#include "primes-1M.inc"
};

__managed__ u64 foundFactor;  // If a factor k is found, save it here.
DEVICE u32 invTab[NPRIMES];
DEVICE int btcTab[NPRIMES];

// returns x * x; 6 MULs.
DEVICE U4 square(U2 x) {
  u32 a, b, c, d;
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

// X at most 79bits. Computes x * x; 11 MULs.
DEVICE U5 square(U3 x) {
  assert(!(x.c & 0xffff8000));
  U2 ab = {x.a, x.b};
  U4 ab2 = square(ab);
  // U3 abc = mul(ab, x.c + x.c);
  U3 abc = ab * (x.c + x.c);
  
  u32 c, d, e;
  asm(
      "add.cc.u32  %0, %3, %5;"
      "addc.cc.u32 %1, %4, %6;"
      "madc.lo.u32 %2, %8, %8, %7;"
      : "=r"(c), "=r"(d), "=r"(e)
      : "r"(ab2.c), "r"(ab2.d), "r"(abc.a), "r"(abc.b), "r"(abc.c), "r"(x.c));
  assert(!(e & 0xc0000000));
  return (U5) {ab2.a, ab2.b, c, d, e};
}

// X at most 79bits. 11MULs.
/*
DEVICE U5 square80(U3 x) {
  assert(!(x.c & 0xffff8000));
  u32 a, b, c, d, e;
  asm("{\n\t"
      ".reg .u32 a2;\n\t"

      "mul.lo.u32     %0, %5, %5;\n\t"
      "mul.lo.u32     %1, %5, %6;\n\t"
      "mul.hi.u32     %2, %5, %6;\n\t"

      "add.u32        a2, %7, %7;\n\t"

      "add.cc.u32     %1, %1, %1;\n\t"
      "addc.cc.u32    %2, %2, %2;\n\t"
      "madc.hi.u32    %3, %5, a2, 0;\n\t"

      "mad.hi.cc.u32  %1, %5, %5, %1;\n\t"
      "madc.lo.cc.u32 %2, %6, %6, %2;\n\t"
      "madc.hi.cc.u32 %3, %6, %6, %3;\n\t"
      "madc.lo.u32    %4, %7, %7, 0;\n\t"
      
      "mad.lo.cc.u32  %2, %5, a2, %2;\n\t"
      "madc.lo.cc.u32 %3, %6, a2, %3;\n\t"
      "madc.hi.u32    %4, %6, a2, %4;\n\t"
      "}"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d), "=r"(e)
      : "r"(x.a), "r"(x.b), "r"(x.c));
  return (U6) {a, b, c, d, e, 0};
}
*/

/*
DEVICE U5 modStep(U5 t, U3 m, u32 R, int sh, int bits) {
  u32 n = mulhi(shl(t.c, t.d, 32 - bits), R);
  t = (U5){0, t.a, t.b, t.c, t.d} - (_U5(m * n) << (sh + bits));
  assert(!t.e && !(t.d & (0xfffffff8 << bits)));
  return t;
}
*/

// m at most 78 bits.
DEVICE U3 mod(U5 x, U3 m, u32 R, int b) {  
  u32 n = mulhi(x.e, R);
  U3 t = mulLow(m, n);
  x = x << (b - 1);
  U3 xup = (U3) {x.c, x.d, x.e} - t;
  assert(!(xup.c >> (b + 2)));
  
  n = mulhi(shr(xup.b, xup.c, b + 2), R);
  t = mulLow(m, n);
  x = (U5){x.a, x.b, xup.a, xup.b, xup.c} << 29;
  xup = (U3){x.c, x.d, x.e} - t;
  if ((xup.c >> (b + 2))) {
    printf("%08x%08x%08x %08x%08x%08x %08x%08x%08x %u\n", x.c, x.d, x.e, xup.a, xup.b, xup.c, m.a, m.b, m.c, n);
  }
  
  assert(!(xup.c >> (b + 2)));

  U4 tmp = (U4){x.b, xup.a, xup.b, xup.c} << (36 - b);
  xup = (U3) {tmp.b, tmp.c, tmp.d};
  n = mulhi(xup.c, R) >> (30 - b);
  t = mulLow(m, n);
  xup = xup - t;
  assert(!(xup.c >> (b + 1)));
  return xup;
}

// Returns the position of the most significant bit that is set.
DEVICE int bfind(u32 x) { int r; asm("bfind.u32 %0, %1;": "=r"(r): "r"(x)); return r; }

// returns 2^exp % m
DEVICE U3 expMod(u32 exp, U3 m) {
  assert(exp & 0x80000000);
  int sh = exp >> 24;
  assert(sh >= 128 && sh < 256);
  if (sh < 160) {
    exp <<= 8;
  } else {
    sh >>= 1;
    exp <<= 7;
  }

  assert(m.c && !(m.c & 0xffffc000));
  int b = bfind(m.c);
  assert(b >= 4);
  u32 R = 0xffffffffffffffffULL / ((0x100000000ULL | shl(m.b, m.c, 32 - b)) + 1);
  
  U3 a = mod((U5){0, 0, 0, 1 << (sh - 96), 1 << (sh - 128)}, m, R, b);
  do {
    a = mod(square(a), m, R, b);
    if (exp & 0x80000000) { a <<= 1; }  // Alternative: a <<= exp >> 31;
  } while (exp += exp);
  return a;
}

DEVICE u32 modInv32(u64 step, u32 prime) {
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

// 3 times 64bit modulo, expensive!
DEVICE int bitToClear(u32 exp, u64 k, u32 prime, u32 inv) {
  u32 kmod = k % prime;
  u32 qmod = (kmod * (u64) (exp << 1) + 1) % prime;
  return (prime - qmod) * (u64) inv % prime;
}

__global__ void __launch_bounds__(1024) initInvTab(u32 exp) {
  assert(gridDim.x * blockDim.x == NPRIMES);
  u32 id = blockIdx.x * blockDim.x + threadIdx.x;
  invTab[id] = modInv32(2 * NCLASS * (u64) exp, primes[id]);
}

__global__ void __launch_bounds__(1024) initBtcTab(u32 exp, u64 k) {
  assert(gridDim.x * blockDim.x == NPRIMES);
  u32 id = blockIdx.x * blockDim.x + threadIdx.x;
  btcTab[id] = bitToClear(exp, k, primes[id], invTab[id]);
}

// 128 blocks, times an internal repeat of 32, times shared memory per block of 24KB, times 8 bits per byte == 3*2^28
#define SIEV_BITS (128 * 32 * 24 * 1024 * 8)
#define SIEV_REPEAT 64
#define SIEV_BLOCKS (SIEV_BITS / (SIEV_REPEAT * NBITS))
#define TEST_REPEAT 256

// Less than 20% of bits survive sieving.
#define KTAB_SIZE (SIEV_BITS / 5)
DEVICE u32 kTabA[KTAB_SIZE];
DEVICE u32 kTabB[KTAB_SIZE];
__managed__ int kTabSizeA;
__managed__ int kTabSizeB;

DEVICE void test(u32 doubleExp, u32 flushedExp, u64 k, u32 *kTab) {
  int n = TEST_REPEAT;
  int pos = TEST_THREADS * TEST_REPEAT * blockIdx.x + threadIdx.x;
  do {
    u32 delta = kTab[pos];
    U3 kk = _U2(k + delta) * doubleExp;
    ++kk.a;
    //incMul(_U2(k + delta), doubleExp)
    U3 r = expMod(flushedExp, kk);
    if (r.a == 1 && !(r.b | r.c)) {
      foundFactor = k;
    }
    pos += TEST_THREADS;
  } while (--n);
}

__global__ void __launch_bounds__(TEST_THREADS, 4) testA(u32 doubleExp, u32 flushedExp, u64 k) {
  test(doubleExp, flushedExp, k, kTabA);
}

__global__ void __launch_bounds__(TEST_THREADS, 4) testB(u32 doubleExp, u32 flushedExp, u64 k) {
  test(doubleExp, flushedExp, k, kTabB);
}

DEVICE void sieve(int *pSize, u32 *kTab) {
  __shared__ u32 words[NWORDS];
  const int tid = threadIdx.x;
  int rep = SIEV_REPEAT;
  do {
    for (int i = 0; i < NWORDS / SIEV_THREADS; ++i) { words[tid + i * SIEV_THREADS] = 0; }
    __syncthreads();
    for (int i = tid; i < NPRIMES; i += SIEV_THREADS) {
      int prime = primes[i];
      int btc0  = btcTab[i];
      int btcAux = btc0 - (NCLASS * NBITS % prime) * blockIdx.x % prime;
      int btc = (btcAux < 0) ? btcAux + prime : btcAux;
      while (btc < NBITS) {
        atomicOr(words + (btc >> 5), 1 << (btc & 0x1f));
        btc += prime;
      }
    }
    __syncthreads();
    
    int popc = 0;
    
    // for (int i = 0, idx = threadIdx.x; i < NWORDS / SIEVE_THREADS; ++i, idx += SIEVE_THREADS) { popc += __popc(words[idx] = ~words[idx]); }
    for (int i = tid; i < NWORDS; i += SIEV_THREADS) { popc += __popc(words[i] = ~words[i]); }
  
    u32 bits = words[tid];
    if (tid < 32) {
      words[0] = 0;
      words[1] = 0xffffffff;
    }
    __syncthreads();
    u32 pos = atomicAdd(words, popc | (1 << 20));
    atomicMin(words + 1, popc);
    __syncthreads();
    if (tid == 0) {
      words[0] = atomicAdd(pSize, words[0] & 0xfffff);
    }
    int min = words[1];
    pos = (pos & 0xfffff) - min * (pos >> 20);
    __syncthreads();
    int p = words[0] + tid;
    int i = tid;
    u32 delta = (tid + blockIdx.x * NWORDS) * 32;
    do {
      while (!bits) {
        bits = words[i += SIEV_THREADS];
        delta += SIEV_THREADS * 32;
      }
      int bit = bfind(bits);
      bits &= ~(1 << bit);
      kTab[p] = delta + bit;
      p += SIEV_THREADS;
    } while (--min);
    p += -tid + (int)pos;
    while (true) {
      while (!bits) {
        i += SIEV_THREADS;
        if (i >= NWORDS) { goto out; }
        bits = words[i];
        delta += SIEV_THREADS * 32;
      }
      int bit = bfind(bits);
      bits &= ~(1 << bit);
      kTab[p++] = delta + bit;
    }
  out:;
  } while (--rep);
}

__global__ void __launch_bounds__(SIEV_THREADS) sievA() { sieve(&kTabSizeA, kTabA); }
__global__ void __launch_bounds__(SIEV_THREADS) sievB() { sieve(&kTabSizeB, kTabB); }

int classTab[NGOODCLASS];

void initClasses(u32 exp) {
  int nClass = 0;
  for (int c = 0; c < NCLASS; ++c) {
    if (acceptClass(exp, c)) {
      classTab[nClass++] = c;
    }
  }
  assert(nClass == NGOODCLASS);
}

u64 calculateK(u32 exp, int bits) {
  return (((u128) 1) << (bits - 1)) / exp;
}

#define CUDA_CHECK_ERR  err = cudaGetLastError(); if (err) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 0; }

int testBlocks(int kSize) {
  return kSize / (TEST_THREADS * TEST_REPEAT); // FIXME round up instead of down.
}

int main() {
  assert(NPRIMES % 1024 == 0);
  
  cudaError_t err;
  // cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  // cudaSetDevice(1);
  CUDA_CHECK_ERR;

  int p1=-1, p2=-1;
  cudaDeviceGetStreamPriorityRange(&p1, &p2);
  CUDA_CHECK_ERR;
  printf("Priority %d %d %d\n", p1, p2, SIEV_BLOCKS);
  
  cudaStream_t sieveStream, testStream;
  cudaStreamCreateWithPriority(&sieveStream, cudaStreamNonBlocking, 0);
  CUDA_CHECK_ERR;
  cudaStreamCreateWithPriority(&testStream, cudaStreamNonBlocking, 1);
  CUDA_CHECK_ERR;
  
  
  const u32 exp = 119904229;
  u64 t1 = timeMillis();
  u64 t0 = t1;
  initClasses(exp);
  printf("initClasses: %llu ms\n", timeMillis() - t1);

  int startPow2 = 69;
  u64 kStart = calculateK(exp, startPow2);
  u64 kEnd   = calculateK(exp, startPow2 + 1);
  u64 k0Start = kStart - (kStart % NCLASS);
  u64 k0End   = kEnd + (NCLASS - (kEnd % NCLASS)) % NCLASS;
  u32 blockSize = NBITS * NCLASS;
  u32 blocks = (kEnd - k0Start + (blockSize - 1)) / blockSize;
  u32 flushedExp = exp << __builtin_clz(exp);
  u32 doubleExp = exp + exp;
  
  printf("exp %u kStart %llu kEnd %llu k0Start %llu k0End %llu, %llu blocks %u, actual %u\n",
         exp, kStart, kEnd, k0Start, k0End, (k0Start + blocks * (u64) blockSize), blocks, 512 * 3 * 1024 / NWORDS);
  
  t1 = timeMillis();
  initInvTab<<<NPRIMES/1024, 1024>>>(exp);
  cudaDeviceSynchronize();
  printf("initInvTab: %llu ms\n", timeMillis() - t1);
  t1 = timeMillis();
  kTabSizeA = 0;
  kTabSizeB = 0;
  
  for (int cid = 0; cid < NGOODCLASS; ++cid) {
    int c = classTab[cid];
    u64 k = k0Start + c;

    int sizeB = kTabSizeB;
    int testBlocksB = testBlocks(sizeB);
    kTabSizeB = 0;
    initBtcTab<<<NPRIMES/1024, 1024>>>(exp, k);
    // if (!cid)
    sievA<<<SIEV_BLOCKS, SIEV_THREADS, 0, sieveStream>>>();
    usleep(100);
    testB<<<testBlocksB, TEST_THREADS, 0, testStream>>>(doubleExp, flushedExp, k);
    cudaDeviceSynchronize();
    int sizeA = kTabSizeA;
    int testBlocksA = testBlocks(sizeA);
    kTabSizeA = 0;
    // if (!cid)
    sievB<<<SIEV_BLOCKS, SIEV_THREADS, 0, sieveStream>>>();
    usleep(100);
    testA<<<testBlocksA, TEST_THREADS, 0, testStream>>>(doubleExp, flushedExp, k);
    cudaDeviceSynchronize();
    u64 t2 = timeMillis();
    printf("%5d: class %5d: %llu; A %d (%d), B %d (%d)\n", cid, c, t2 - t1, sizeA, testBlocksA, sizeB, testBlocksB);
    t1 = t2;    
    // if (foundFactor) { printf("Factor K: %llu\n", foundFactor); break; }
  }
  printf("Total time: %llu ms\n", timeMillis() - t0);
  cudaStreamDestroy(sieveStream);
  cudaStreamDestroy(testStream);
  CUDA_CHECK_ERR;
  // cudaDeviceReset();
}


/*

    initBtcTab<<<NPRIMES/1024, 1024>>>(exp, k);
    sieveAndTest<<<SIEV_BLOCKS, SIEV_THREADS>>>(doubleExp, flushedExp, k);
    cudaDeviceSynchronize();
    u64 t2 = timeMillis();
    printf("%llu\n", t2 - t1);
    t1 = t2;

  __global__ void __launch_bounds__(SIEV_THREADS, 4) sieveAndTest(u32 doubleExp, u32 flushedExp, u64 k) {
  __shared__ u32 words[NWORDS];
  const int tid = threadIdx.x;
  int rep = SIEV_REPEAT;
  while (true) {
    for (int i = 0; i < NWORDS / SIEV_THREADS; ++i) { words[tid + i * SIEV_THREADS] = 0; }
    __syncthreads();
    for (int i = tid; i < NPRIMES; i += SIEV_THREADS) {
      int prime = primes[i];
      int btc0  = btcTab[i];
      int btcAux = btc0 - (NCLASS * NBITS % prime) * blockIdx.x % prime;
      int btc = (btcAux < 0) ? btcAux + prime : btcAux;
      while (btc < NBITS) {
        atomicOr(words + (btc >> 5), 1 << (btc & 0x1f));
        btc += prime;
      }
    }
    __syncthreads();    
    u32 bits = ~words[tid];
    int i = tid;
    u32 delta = (tid + blockIdx.x * NWORDS) * 32;
    while (true) {
      while (!bits) {
        i += SIEV_THREADS;
        if (i >= NWORDS) { goto out; }
        bits = ~words[i];
        delta += SIEV_THREADS * 32;
      }
      int bit = bfind(bits);
      bits &= ~(1 << bit);
      U3 r = expMod(flushedExp, incMul(_U2(k + (delta + bit)), doubleExp));
      if (r.a == 1 && !(r.b | r.c)) { foundFactor = k; }
    }
  out:
    if (!--rep) { break; }
    __syncthreads();
  }
}
*/

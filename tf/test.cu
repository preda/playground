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

typedef unsigned short u16;
typedef unsigned u32;
typedef unsigned long long u64;
typedef __uint128_t u128;

// Multi-precision unsigned ints with the given number of words.
// The least-significant word is "a".
struct U2 { u32 a, b; };
struct U3 { u32 a, b, c; };
struct U4 { u32 a, b, c, d; };
struct U5 { u32 a, b, c, d, e; };
struct U6 { u32 a, b, c, d, e, f; };

#define ASIZE(a) (sizeof(a) / sizeof(a[0]))

// Threads per block, doing sieving and factor-testing.
#define THREADS_PER_BLOCK (512 + 128)
// How many threads do sieving, out of THREADS_PER_BLOCK.
#define SIEVE_THREADS 128
// Threads for doing the BTC init, done only once per exponent, at the beginning.
#define BTC_THREADS 1024

// How many words of shared memory to use for sieving.
#define NWORDS (12 * 1024)

// Must update acceptClass() when changing these.
#define NCLASS     (4 * 3 * 5 * 7 * 11 * 13)

// Derived values below.
// How many threads do "modular exponentiation" to test factor candidates.
#define WORK_THREADS (THREADS_PER_BLOCK - SIEVE_THREADS)
// Bits for sieving.
#define NBITS (NWORDS << 5)
// Number of pre-computed primes for sieving.
#define NPRIMES (ASIZE(primes))
// Out of NCLASS, how many classes pass acceptClass().
#define NGOODCLASS (2 * 2 * 4 * 6 * 10 * 12)
// How many blocks for BTC init.
#define BTC_BLOCKS (NPRIMES / BTC_THREADS)
// Block for sieving+testing.
#define BLOCKS 32


// Returns whether 2 * c * exp + 1 is 1 or 7 modulo 8.
// Any Marsenne factor must be of this form. See http://www.mersenne.org/various/math.php
bool q1or7mod8(u32 exp, u32 c) {
  return !(c & 3) || ((c & 3) + (exp & 3) == 4);
}

// whether 2 * c * exp + 1 != 0 modulo prime
bool notMultiple(u32 exp, u32 c, unsigned prime) {
  return (2 * c * (u64) exp + 1) % prime;
}

bool acceptClass(u32 exp, u32 c) {
  // Keep in sync with NCLASS
  return q1or7mod8(exp, c) && notMultiple(exp, c, 3) && notMultiple(exp, c, 5)
    && notMultiple(exp, c, 7) && notMultiple(exp, c, 11) && notMultiple(exp, c, 13);
}

u64 timeMillis() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

__device__ const u32 primes[] = {
#include "primes-1M-17.inc"
};

__managed__ u64 foundFactor;  // If a factor k is found, save it here.
__managed__ u16 classTab[NGOODCLASS];  // The class value for each "good" class.

// BTC means "bit to clear", the first bit position to clear in the big bit block when sieving.
// Keeps the BTC for each prime, for each good class. The BTC is updated during sieving.
// This is the bulk of the GPU [global] memory usage.
// __device__ u32 classBtcTab[NGOODCLASS][NPRIMES];
__device__ u32 invTab[NPRIMES];

// Funnel shift left.
__device__ u32 shl(u32 a, u32 b, int n) {
  u32 r;
  asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(n));
  return r;
}

// Funnel shift right.
__device__ u32 shr(u32 a, u32 b, int n) {
  u32 r;
  asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(n));
  return r;
}

__device__ U5 add(U5 x, U3 y) {
 u32 a, b, c, d, e;
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
 u32 a, b, c, d;
  asm("add.cc.u32  %0, %4, %8;"
      "addc.cc.u32 %1, %5, %9;"
      "addc.cc.u32 %2, %6, %10;"
      "addc.u32    %3, %7, 0;"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(x.d), "r"(y.a), "r"(y.b), "r"(y.c));
  return (U4) {a, b, c, d};
}

__device__ U3 add(U3 x, U3 y) {
  u32 a, b, c;
  asm("add.cc.u32  %0, %3, %6;"
      "addc.cc.u32 %1, %4, %7;"
      "addc.u32    %2, %5, %8;"
      : "=r"(a), "=r"(b), "=r"(c)
      : "r"(x.a), "r"(x.b), "r"(x.c),
        "r"(y.a), "r"(y.b), "r"(y.c));
  return (U3) {a, b, c};
}

__device__ static U4 sub(U4 x, U4 y) {
  u32 a, b, c, d;
  asm("sub.cc.u32  %0, %4, %8;"
      "subc.cc.u32 %1, %5, %9;"
      "subc.cc.u32 %2, %6, %10;"
      "subc.u32    %3, %7, %11;"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(x.d),
        "r"(y.a), "r"(y.b), "r"(y.c), "r"(y.d));
  return (U4) {a, b, c, d};
}

__device__ static U5 sub(U5 x, U5 y) {
  u32 a, b, c, d, e;
  asm("sub.cc.u32  %0, %5, %10;"
      "subc.cc.u32 %1, %6, %11;"
      "subc.cc.u32 %2, %7, %12;"
      "subc.cc.u32 %3, %8, %13;"
      "subc.u32    %4, %9, %14;"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d), "=r"(e)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(x.d), "r"(x.e),
        "r"(y.a), "r"(y.b), "r"(y.c), "r"(y.d), "r"(y.e));
  return (U5) {a, b, c, d, e};
}

// returns x - (y << 32)
__device__ U5 subShl1w(U5 x, U4 y) {
  U4 t = sub((U4) {x.b, x.c, x.d, x.e}, y);
  return (U5) {x.a, t.a, t.b, t.c, t.d};
}

// returns x * n; 4 MULs.
__device__ U3 mul(U2 x, u32 n) {
  u32 a, b, c;
  asm(
      "mul.hi.u32     %1, %3, %5;"
      "mul.lo.u32     %0, %3, %5;"
      "mad.lo.cc.u32  %1, %4, %5, %1;"
      "madc.hi.u32    %2, %4, %5, 0;"
      : "=r"(a), "=r"(b), "=r"(c)
      : "r"(x.a), "r"(x.b), "r"(n));
  return (U3) {a, b, c};
}

// returns x * n; 6 MULs.
__device__ U4 mul(U3 x, u32 n) {
  u32 a, b, c, d;
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

// returns (x*n >> 32) + (n ? 1 : 0). Used for Montgomery reduction. 5 MULs.
__device__ U3 mulM(U3 x, u32 n) {
  u32 a, b, c;
  asm(
      "mul.lo.u32     %1, %5, %6;"
      "mad.hi.u32     %0, %3, %6, %7;"
      "madc.hi.cc.u32 %1, %4, %6, %1;"
      "madc.hi.u32    %2, %5, %6, 0;"
      : "=r"(a), "=r"(b), "=r"(c)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(n), "r"(x.b * n + (n ? 1 : 0)));
  return (U3) {a, b, c};
}

// returns x * x; 6 MULs.
__device__ U4 square(U2 x) {
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

// returns x * x; 12 MULs.
__device__ U6 square(U3 x) {
  U2 ab = {x.a, x.b};
  U4 ab2 = square(ab);
  U3 abc = mul(ab, x.c + x.c);
  
  u32 c, d, e, f;
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

__device__ U6 makeU6(U3 x) { return (U6) {x.a, x.b, x.c, 0, 0, 0}; }
__device__ U5 makeU5(U4 x) { return (U5) {x.a, x.b, x.c, x.d, 0}; }
__device__ U6 makeU6(U5 x) { return (U6) {x.a, x.b, x.c, x.d, x.e, 0}; }
__device__ U6 makeU6(U4 x) { return makeU6(makeU5(x)); }
__device__ U2 makeU2(u64 x) { return (U2) {(u32) x, (u32) (x >> 32)}; }

__device__ U3 shl(U3 x, int n) {
  assert(n >= 0 && n < 32 && !(x.c >> (32 - n)));
  return (U3) {x.a << n, shl(x.a, x.b, n), shl(x.b, x.c, n)};
}

__device__ U4 shl(U4 x, int n) {
  // assert(n >= 0 && n < 32 && !(x.d >> (32 - n)));
  return (U4) {x.a << n, shl(x.a, x.b, n), shl(x.b, x.c, n), shl(x.c, x.d, n)};
}

__device__ U5 shl(U5 x, int n) {
  assert(n >= 0 && n < 32 && !(x.e >> (32 - n)));
  U4 t = shl((U4) {x.a, x.b, x.c, x.d}, n);
  return (U5) {t.a, t.b, t.c, t.d, shl(x.d, x.e, n)};
}

__device__ U3 mod(U5 x, U3 m) {
  assert(m.c);
  int sh = __clz(m.c) + 1;
  if (sh > 26) {
    m = shl(m, sh - 26);
    sh = 26;
  }
  assert(sh >= 3 && sh <= 26);
  u32 R = 0xffffffffffffffffULL / ((0x100000000ULL | shl(m.b, m.c, sh)) + 1);
  u32 n = mulhi(x.e, R);
  U4 t = sub((U4) {x.b, x.c, x.d, x.e}, shl(mul(m, n), sh));
  x = (U5){x.a, t.a, t.b, t.c, t.d};
  assert(!(x.e & 0xfffffff8));
  n = mulhi(shl(x.d, x.e, 29), R);
  U5 mn = shl(makeU5(mul(m, n)), sh + 3);
  x = sub(x, mn);
  assert(!x.e && !(x.d & 0xffffffc0));
  n = mulhi(shl(x.c, x.d, 26), R) >> (26 - sh);
  t = sub((U4) {x.a, x.b, x.c, x.d}, mul(m, n));
  assert(!t.d);
  assert(!(t.c >> (35 - sh)));
  return (U3) {t.a, t.b, t.c};
}

// Compute m' such that: (u32) (m * m') == 0xffffffff, using extended binary euclidian algorithm.
// See http://www.ucl.ac.uk/~ucahcjm/combopt/ext_gcd_python_programs.pdf
// m is odd.
__device__ static u32 mprime(u32 m) {
  m = (m >> 1) + 1;
  u32 u = m;
  u32 v = m << 31; 
  for (int i = 0; i < 30; ++i) {
    u = (u >> 1) + ((u & 1) ? m : 0);
    v = shr(v, u, 1);
  }
  return v | 1;
}

// Montgomery Reduction
// See https://www.cosic.esat.kuleuven.be/publications/article-144.pdf
// Returns x * U^-1 mod m
__device__ static U3 montRed(U6 x6, U3 m, u32 mp) {
  assert(!(x6.f & 0xc0000000));
  assert(x6.a + (x6.a * mp * m.a) == 0);
  U5 x5 = add((U5) {x6.b, x6.c, x6.d, x6.e, x6.f}, mulM(m, x6.a * mp));
  U4 x4 = add((U4) {x5.b, x5.c, x5.d, x5.e}, mulM(m, x5.a * mp));
  U3 x3 = add((U3) {x4.b, x4.c, x4.d}, mulM(m, x4.a * mp));
  assert(!(x3.c & 0xc0000000));
  return x3;
}

// returns 2^exp % m
__device__ U3 expMod(u32 exp, U3 m) {
  assert(exp & 0x80000000);
  u32 mp = mprime(m.a);

  int sh = exp >> 26;
  assert(sh >= 32 && sh < 64);
  U3 a = mod((U5){0, 0, 0, 0, 1 << (sh - 32)}, m);
  for (exp <<= 6; exp; exp += exp) {
    U6 a2 = square(a);
    a = montRed(a2, m, mp);
    if (exp & 0x80000000) { a = shl(a, 1); }
  }
  return montRed(makeU6(a), m, mp);
}

// return 2 * k * p + 1 as U3
__device__ U3 makeQ(u32 p, u64 k) {
  return add(mul(makeU2(k), p + p), (U3){1, 0, 0});
}

// returns whether (2*k*p + 1) is a factor of (2^p - 1)
__device__ bool isFactor(u32 exp, u32 flushedExp, u64 k) {
  U3 q = makeQ(exp, k);
  U3 r = expMod(flushedExp, q);
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

// 3 times 64bit modulo, expensive!
__device__ int bitToClear(u32 exp, u64 k, u32 prime, u32 inv) {
  u32 kmod = k % prime;
  u32 qmod = (kmod * (u64) (exp << 1) + 1) % prime;
  return (prime - qmod) * (u64) inv % prime;
}

// Returns the position of the most significant bit that is set.
__device__ int bfind(u32 x) { int r; asm("bfind.u32 %0, %1;": "=r"(r): "r"(x)); return r; }

// Initializes the classBtcTab array in GPU memory.
__global__ void __launch_bounds__(1024) initInvTab(u32 exp, u64 k0) {
  const u64 step = 2 * NCLASS * (u64) exp;
  for (const u32 *p = primes + ID, *end = primes + NPRIMES; p < end; p += gridDim.x * blockDim.x) {
    invTab[p - primes] = modInv32(step, *p);
    /*
    u32 inv = modInv32(step, prime);
    for (int i = 0; i < NGOODCLASS; ++i) {
      classBtcTab[i][p - primes] = bitToClear(exp, k0 + classTab[i], prime, inv);
    }
    */
  }
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK, 2) tf(u32 exp, u64 k0, int c0, int nRounds) {
  __shared__ u32 words[NWORDS];
#define LOCAL_WORDS ((NWORDS + WORK_THREADS - 1) / WORK_THREADS)
  u32 localWords[LOCAL_WORDS];
  localWords[0] = 0;
  u32 *localEnd = localWords;
  
  const int tid = threadIdx.x;
  const int cid = c0 + blockIdx.x;
  const int c = classTab[cid];
  const u32 flushedExp = exp << __clz(exp);
  // u32 * const btcTab = classBtcTab[cid];
  u64 kBlock = k0 + c + (tid - SIEVE_THREADS) * (32 * NCLASS) - NBITS * (u64) NCLASS;
  bool shouldExit = false;
  // u32 timeSieve, timeTest, timeCopy;

  for (int round = 0; round < nRounds && !shouldExit; ++round) {
    // u32 time0 = clock();
    if (tid < SIEVE_THREADS) {
      // u32 *btcp = btcTab + tid;
      for (const u32 *p = primes + tid, *end = primes + NPRIMES, *invp = invTab; p < end;
           p += SIEVE_THREADS, invp += SIEVE_THREADS) {
        int prime = *p;
        int btc = bitToClear(exp, k0 + c, prime, *invp);
        // int btc = *btcp;
        while (btc < NBITS) {
          atomicOr(words + (btc >> 5), 1 << (btc & 0x1f));
          btc += prime;
        }
        // *btcp = btc - NBITS;
      }
      // timeSieve = clock() - time0;
    } else {
      u32 *p = localWords;
      u32 bits = *p;
      u64 kWord = kBlock;
      while (true) {
        while (!bits) {
          kWord += WORK_THREADS * (32 * NCLASS);
          if (++p >= localEnd /*|| kWord >= kEnd*/) { goto out; }
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
    // int pops = 0;
    if (tid >= SIEVE_THREADS) {
      localEnd = localWords;
      // u32 *out = localWords;
      for (u32 *p = words + (tid - SIEVE_THREADS), *end = words + NWORDS; p < end; p += WORK_THREADS) {
        u32 save = ~*p;
        *p = 0;
        *localEnd++ = save;
        // pops += __popc(save);
      }
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
    kBlock += NBITS * (u64) NCLASS;
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
  return (((u128) 1) << (bits - 1)) / exp;
  // return k - k % NCLASS;
}

#define CUDA_CHECK_ERR  err = cudaGetLastError(); if (err) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 0; }

int main() {
  assert(NPRIMES % BTC_THREADS == 0);
  assert(NGOODCLASS % BLOCKS == 0);
  
  cudaError_t err;
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  // cudaSetDevice(1);
  CUDA_CHECK_ERR;
  
  const u32 exp = 119904229;
  initClasses(exp);

  int startPow2 = 67;
  u64 kStart = calculateK(exp, startPow2);
  u64 kEnd   = calculateK(exp, startPow2 + 1);
  u64 k0Start = kStart - (kStart % NCLASS);
  u64 k0End   = kEnd + (NCLASS - (kEnd % NCLASS)) % NCLASS;
  u64 perRound = NBITS * (u64) NCLASS;
  u64 rounds = (kEnd - k0Start + (perRound - 1)) / perRound;

  printf("exp %u kStart %llu kEnd %llu k0Start %llu k0End %llu, %llu rounds %llu\n",
         exp, kStart, kEnd, k0Start, k0End, (k0Start + rounds * perRound), rounds);
  
  u64 t1 = timeMillis();
  initInvTab<<<BTC_BLOCKS, BTC_THREADS>>>(exp, k0Start);
  cudaDeviceSynchronize();
  u64 t2 = timeMillis();
  printf("initBtcTab with %lu blocks: %llu ms\n", BTC_BLOCKS, (t2 - t1));
  CUDA_CHECK_ERR;

  u64 t3 = timeMillis();
  for (int cid = 0; cid < NGOODCLASS; cid += BLOCKS) {
    tf<<<BLOCKS, THREADS_PER_BLOCK>>>(exp, k0Start, cid, rounds);
    cudaDeviceSynchronize();
    u64 t4 = timeMillis();
    printf("class %d time %llu\n", cid, (t4 - t3));
    t3 = t4;
    CUDA_CHECK_ERR;
  }
  printf("TF: %llu\n", (t3 - t2));
  CUDA_CHECK_ERR;
  if (foundFactor) {
    printf("Found factor %llu\n", foundFactor);
  }
  // cudaDeviceReset();
  return 0;
}


/*
  // if (x.d & 0xffffffc0) { printf("%08x%08x%08x%08x%08x %08x%08x%08x %u %d\n", s.e, s.d, s.c, s.b, s.a, m.c, m.b, m.a, n, sh); }
  // if (t.d) { printf("%08x%08x%08x%08x %08x%08x%08x %u %d\n", x.d, x.c, x.b, x.a, m.c, m.b, m.a, n, sh); }
  
  __device__ U3 mod(U5 x, U3 m) {
  assert(m.c);
  int shift = __clz(m.c) + 1;
  assert(shift >= 18);
  // R is an approximation from below of 2^64 / (m << shift).
  u32 R = 0xffffffffffffffffULL / ((0x100000000ULL | shl(m.b, m.c, shift)) + 1);
  // A faster but lower-quality approximation of the same.
  // u32 R = 0x7fffffff00000001ULL / shl(m.b, m.c, shift - 1);
  
  u32 n;  
  n = mulhi(x.e, R);
  x = subShl1w(x, shl(mul(m, n), shift));
  assert(!(x.e & 0xfffffff0));
  x = shl(x, 28);
  
  n = mulhi(x.e, R);
  x = subShl1w(x, shl(mul(m, n), shift));
  assert(!(x.e & 0xfffffff0));
  x = shl(x, 28);
  
  n = mulhi(x.e, R) & 0xffffffc0;
  U5 s = x;
  x = subShl1w(x, shl(mul(m, n), shift));
  if (x.e & 0xffffff80) {
    printf("%08x%08x%08x%08x%08x %08x%08x%08x%08x%08x %08x%08x%08x %08x %08x\n",
           s.e, s.d, s.c, s.b, s.a, x.e, x.d, x.c, x.b, x.a, m.c, m.b, m.a, R, n);
  }
  assert((!(x.e & 0xffffff80)));
  x = shl(x, 8);
  assert(!(x.e & 0xffff8000));
  return (U3) {x.c, x.d, x.e};
}
*/

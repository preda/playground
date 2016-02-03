// Copyright (c) Mihai Preda, 2015 - 2016

/*
  Aa program for trial factoring of Mersenne numbers on a CUDA GPU.

  Mersenne numbers are of the form 2**exp - 1; see http://www.mersenne.org/various/math.php
  This is inpired by mfaktc: http://www.mersenneforum.org/mfaktc/

  For a given mersenne number 2**exp-1, where exp is prime, the factors are of the form
  m = 2*k*exp + 1, and we're interested only in prime factors.

  Limits: exp < 2**31; 2**64 < m < 2**76.

  First prime candidate factors are generated -- this is called "sieving" because it uses
  Erathostene's sieve. Next each candidate m is tested by the computing the modular
  exponentiation reminder r = 2**exp modulo m. If this reminder is equal to 1, it means
  that m is a factor of 2^exp-1, and thus the mersenne number is not prime.

  
  Naming conventions used:

  1. type names:
     - u8, u16, u32, u64, u128: unsigned integer with the given number of *bits*.
     - U2, U3, U4, etc: unsigned long integer with the given number of 32-bit words.
       The words of a long integer are named "a", "b", "c", etc, a being the least-significant.
       
  2. operators on long integers:
     - usual: +, -, *.
     - bit shifts: <<, >>.
     - shr1w(): word shift right
     - funnel shift returning one word: shl, shr
     - cast to larger type, e.g. _U4(U3 x)
     - mulLow(): multiplication computing only the lower words
     - shr3wMul(): multiplication computing  only the higher words
     - equality ==
     - square
*/

#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/unistd.h>

#define DEVICE __device__ static
#define ASIZE(a) (sizeof(a) / sizeof(a[0]))

#include "widemath.h"
#include "debug.h"

// Table of small primes.
DEVICE const __restrict__ u32 primes[] = {
#include "primes-1M.inc"
};
// Number of pre-computed primes for sieving.
#define NPRIMES (ASIZE(primes))

// Unit tests. A series of pairs (exponent, k) where k represents a factor.
struct Test { u32 exp; u64 k; };
#include "tests.inc"

// Threads for initBtcTabs()
#define INIT_BTC_THREADS 256
// Threads per sieving block.
#define SIEVE_THREADS 512
// Threads per testing block.
#define TEST_THREADS 512

// How many words of shared memory to use for sieving.
#define NWORDS (8 * 1024)
// Bits for sieving (each word is 32 bits).
#define NBITS (NWORDS << 5)
// How many rows are needed at most in a testing block of TEST_THREADS colums.
#define TEST_ROWS (NBITS / 5 / TEST_THREADS + 1)

// Must update acceptClass() when changing these.
#define NCLASS     (4 * 3 * 5 * 7 * 11)
// Out of NCLASS, how many classes pass acceptClass(). Sync with NCLASS.
#define NGOODCLASS (2 * 2 * 4 * 6 * 10)

// Some powers of 2 as floats, used by inv160()
#define TWO16f  65536.0f
#define TWO17f  131072.0f
#define TWO28f  268435456.0f
#define TWO32f  4294967296.0f
#define TWO64f  18446744073709551616.0f

// Table with inv(exp). Initialized once per exponent.
DEVICE u32 invTab[NPRIMES];

// "Bit to clear" table, depends on exponent, k0, and class; initialized once per exponent.
DEVICE int btcTabs[NGOODCLASS][NPRIMES];

// Sieved Ks. sieve() outputs here, test() reads from here.
DEVICE u32 kTab[NGOODCLASS][NBITS / 5];
// kTabSize[i] has the number of elements in kTab[i]
DEVICE u32 kTabSize[NGOODCLASS];

__managed__ U3 foundFactor; // If a factor m is found, save it here.
DEVICE u64 foundK;     // K for a found factor.
__managed__ int classTab[NGOODCLASS];

// Helper to check and bail out on any CUDA error.
#define CUDA_CHECK  {cudaError_t _err = cudaGetLastError(); if (_err) { printf("CUDA error: %s\n", cudaGetErrorString(_err)); return 0; }}

inline void checkCuda(cudaError_t result) {
  if (result != cudaSuccess) { printf("CUDA Runtime Error: %s\n", cudaGetErrorString(result)); }
}

// Returns whether 2 * c * exp + 1 is 1 or 7 modulo 8.
// Any Marsenne factor must be of this form. See http://www.mersenne.org/various/math.php
bool q1or7mod8(u32 exp, u32 c) { return !(c & 3) || ((c & 3) + (exp & 3) == 4); }

// whether 2 * c * exp + 1 != 0 modulo prime
bool notMultiple(u32 exp, u32 c, unsigned prime) { return (2 * c * (u64) exp) % prime != prime - 1; }
// { return (2 * c * (u64) exp + 1) % prime; }

bool acceptClass(u32 exp, u32 c) {
#define P(p) notMultiple(exp, c, p)
  return q1or7mod8(exp, c) && P(3) && P(5) && P(7) && P(11);
#undef P
}

u64 timeMillis() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

// Returns x % m, given u the "inverse" of m (2**160 / m); m at most 77 bits.
DEVICE U3 mod(U5 x, U3 m, U3 u) {
  return (U3){x.a, x.b, x.c} - mulLow(m, shr3wMul((U3) {x.c, x.d, x.e}, u));
}

// float lower approximation of 2**32 / x
DEVICE float floatInv(U3 x) { return __frcp_rd(__ull2float_ru(_u64(shr1w(x)) + 1)); }

// float lower approximation of a + b * 2**32; (__fmaf_rz(b, TWO32f, a))
DEVICE float floatOf(u32 a, u32 b) { return __ull2float_rz(_u64((U2) {a, b})); }

// float lower approximation of (a + b * 2**32) * nf
DEVICE float floatOf(u32 a, u32 b, float nf) { return __fmul_rz(floatOf(a, b), nf); }

// Returns 2**160 / n
DEVICE U3 inv160(U3 n, float nf) {
  // 1
  assert(nf * TWO64f < TWO32f);
  u32 rc = (u32) __fmul_rz(TWO64f, nf);
  U4 q = shl1w((~mulLow(n, rc)) + 1);

  // 2
  float qf = floatOf(q.c, q.d, nf) * TWO16f;
  assert(qf < TWO28f);
  u32 qi = (u32) qf;
  u32 rb = (qi << 16);
  rc += (qi >> 16);
  q = q - ((n * qi) << 16);
  assert(q.d == 0);

  // 3
  qf = floatOf(q.b, q.c, nf);
  assert(qf < (1 << 24));
  qi = (u32) qf;
  U2 rup = (U2){rb, rc} + qi;
  q = q - n * qi;
  assert(q.d == 0);
  
  // 4
  qf = floatOf(q.b, q.c, nf) * TWO17f;
  assert(qf < (1 << 22));
  qi = (u32) qf;
  rup = rup + (qi >> 17);
  U3 ret = (U3) {(qi << 15), rup.a, rup.b};

  // p("n ", n); p("q ", q);
  q = ((U4) {0, q.a, q.b, q.c}) - ((n * qi) << 15);
  // if (q.d) { printf("qi %d qf %.2f %.10f %.10f %.10f %f nf %f", qi, qf, t1, t2, t3, TWO32f, (nf * TWO64f)); p("q4 ", q); }
  assert(q.d == 0);
  
  // 5
  qf = floatOf(q.b, q.c, nf);
  assert(qf < (1 << 20));
  return ret + (u32) qf;
}

// Returns 2**exp % m
DEVICE U3 expMod(u32 exp, U3 m) {
  assert(exp & 0x80000000);
  assert(m.c && !(m.c & 0xffffc000));
  int sh = exp >> 25;
  assert(sh >= 64 && sh < 128);
  exp <<= 7;

  float nf = floatInv(m);
  U3 u = inv160(m, nf);
  U3 a = mod((U5){0, 0, 1 << (sh - 64), 1 << (sh - 96), 0}, m, u);
  do {
    a = mod(square(a), m, u);
    if (exp & 0x80000000) { a <<= 1; }
  } while (exp += exp);
  a = a - mulLow(m, (u32) floatOf(a.b, a.c, nf));
  return (a.c >= m.c && a == (m + 1)) ? (U3) {1, 0, 0} : a;
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

__global__ void __launch_bounds__(INIT_BTC_THREADS) initBtcTabs(u32 exp, u64 kBase) {
  assert(gridDim.x == NGOODCLASS);
  int *btcTab = btcTabs[blockIdx.x];
  u64 k = kBase + classTab[blockIdx.x];
  // if (!threadIdx.x) { printf("start class %d (%d)\n", classTab[blockIdx.x], blockIdx.x); }
  for (int id = threadIdx.x; id < NPRIMES; id += blockDim.x) {
    btcTab[id] = bitToClear(exp, k, primes[id], invTab[id]);
  }
  // if (!threadIdx.x) { printf("ended class %d (%d)\n", classTab[blockIdx.x], blockIdx.x); }
}

__global__ void __launch_bounds__(TEST_THREADS) test(u32 doubleExp, u32 flushedExp, u64 k0) {
  // if (!threadIdx.x) { printf("Start %d\n", blockIdx.x); }
  U3 m0 = _U2(k0 + classTab[blockIdx.x]) * doubleExp;
  m0.a |= 1;
  int n = 0;
  u32 *kTabBase = kTab[blockIdx.x];
  for (u32 *p = kTabBase + threadIdx.x, *end = kTabBase + kTabSize[blockIdx.x]; p < end; p += blockDim.x) {
    u32 delta = *p;
    U3 r = expMod(flushedExp, m0 + _U2(NCLASS * (u64) delta * doubleExp));
    ++n;
    if (r == (U3) {1, 0, 0}) {
      foundK = k0 + classTab[blockIdx.x] + NCLASS * (u64) delta;
      // printf("factor k: %llu\n", foundK);
      asm("trap;"); // Stop the block and report an error.
    }
  }
  // if (!threadIdx.x) { printf("End %d %d\n", blockIdx.x, n); }
}

__global__ void testSingle(u32 doubleExp, u32 flushedExp, u64 k) {
  U3 m = _U2(k) * doubleExp;
  m.a |= 1;
  U3 r = expMod(flushedExp, m);
  foundFactor = r;
}

// Returns the position of the most significant bit that is set.
// DEVICE int bfind(u32 x) { int r; asm("bfind.u32 %0, %1;": "=r"(r): "r"(x)); return r; }

// Sieve bits using shared memory.
// For each prime from the primes[] table, starting at a position corresponding to a
// multiple of prime ("btc"), periodically set the bit to indicate a non-prime.
__global__ void __launch_bounds__(SIEVE_THREADS) sieve() {
  __shared__ u32 words[NWORDS];

  // Set shared memory to zero.
  for (int i = threadIdx.x; i < NWORDS; i += blockDim.x) { words[i] = 0; }
  __syncthreads();

  // Sieve bits.
  int *btcTab = btcTabs[blockIdx.x];
  for (int i = threadIdx.x; i < NPRIMES; i += blockDim.x) {
    int prime = primes[i];
    int btc = btcTab[i];
    while (btc < NBITS) {
      atomicOr(words + (btc >> 5), 1 << (btc & 0x1f));
      btc += prime;
    }
    btcTab[i] = btc - NBITS;
  }
  __syncthreads();

  u32 bits = ~words[threadIdx.x];
  words[threadIdx.x] = 0;
  __syncthreads();
  
  int popc = __popc(bits);
  for (int i = blockDim.x + threadIdx.x; i < NWORDS; i += blockDim.x) { popc += __popc(~words[i]); }
  u32 *out = kTab[blockIdx.x] + atomicAdd(words, popc);
  __syncthreads();

  if (threadIdx.x == 0) { kTabSize[blockIdx.x] = words[0]; }
  int i = threadIdx.x;
  while (true) {
    while (bits) {
      int bit = __clz(__brev(bits)); // Equivalent to: __ffs(bits) - 1; 
      bits &= bits - 1;  // Equivalent to: bits &= ~(1 << bit); but likely faster
      
      // int bit = bfind(bits);
      // bits &= ~(1 << bit);
      *out++ = (i << 5) + bit;
    }
    if ((i += blockDim.x) >= NWORDS) { break; }
    bits = ~words[i];
  }
}

  /*
  // Copy shared memory to global memory.
  u32 *out = sievedBits[blockIdx.x];
  for (int i = threadIdx.x; i < NWORDS; i += blockDim.x) {
    out[i] = ~words[i];
  }
  */

// Among all the NCLASS classes, select the ones that are "good",
// i.e. not corresponding to a multiple of a small prime.
void initClasses(u32 exp) {
  int nClass = 0;
  for (int c = 0; c < NCLASS; ++c) {
    if (acceptClass(exp, c)) {
      classTab[nClass++] = c;
      // if (c == 992) { printf("class id %d\n", nClass - 1); }
    }
  }
  assert(nClass == NGOODCLASS);
}

// The smallest k that produces a factor m = (2*k*exp + 1) such that m >= 2**bits
u64 calculateK(u32 exp, int bits) { return ((((u128) 1) << (bits - 1)) + (exp - 2)) / exp; }

// Run one unit-test case.
bool testOne(u32 exp, u64 k) {  
  u32 flushedExp = exp << __builtin_clz(exp);
  u32 doubleExp = exp + exp;
  printf("\r%10u %20llu", exp, k);
  testSingle<<<1, 1>>>(doubleExp, flushedExp, k);
  cudaDeviceSynchronize(); CUDA_CHECK;
  if (foundFactor.a != 1 || foundFactor.b || foundFactor.c) {
    printf("ERROR %10u %20llu m ", exp, k);
    return false;
  }
  return true;
}

void time(const char *s = 0) {
  static u64 prev = 0;
  u64 now = timeMillis();
  if (prev && s) {
    printf("%s: %llu ms\n", s, now - prev);
  }
  prev = now;
}

int main(int argc, char **argv) {
  assert(argc > 0);
  assert(NPRIMES % 1024 == 0);
  
  // cudaSetDevice(1);
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync); CUDA_CHECK;
  
  if (argc == 1) {
    printf("Running selftest..\n");
    for (Test *t = tests, *end = tests + ASIZE(tests); t < end; ++t) {
      if (!testOne(t->exp, t->k)) { return -1; }
    }
    printf("\n%lu tests passed ok\n", ASIZE(tests));
    return 0;
  }

  u32 exp = (u32) atol(argv[1]);
  int startPow2 = (argc >= 3) ? atoi(argv[2]) : 65;
  
  time();
  initClasses(exp);
  time("initClasses");

  u64 k0 = calculateK(exp, startPow2);
  k0 -= k0 % NCLASS;
  u64 kEnd = calculateK(exp, startPow2 + 1);
  kEnd += (NCLASS - (kEnd % NCLASS)) % NCLASS;
  u32 kStep = NBITS * NCLASS;
  u32 repeat = (kEnd - k0 + (kStep - 1)) / kStep;
  u32 flushedExp = exp << __builtin_clz(exp);
  u32 doubleExp = exp + exp;
  printf("k range: %llu - %llu. %u iterations.\n", k0, kEnd, repeat);
    
  initInvTab<<<NPRIMES/1024, 1024>>>(exp);
  CUDA_CHECK; cudaDeviceSynchronize(); time("initInvTab");
  initBtcTabs<<<NGOODCLASS, INIT_BTC_THREADS>>>(exp, k0);
  CUDA_CHECK; cudaDeviceSynchronize(); time("initBtcTabs");
  
  for (int i = 0; i < 40; ++i, k0 += kStep) {
    sieve<<<NGOODCLASS, SIEVE_THREADS>>>();
    cudaDeviceSynchronize(); CUDA_CHECK;
    // time("Sieve");

    test<<<NGOODCLASS, TEST_THREADS>>>(doubleExp, flushedExp, k0);
    cudaDeviceSynchronize(); CUDA_CHECK;
    // time("Test");

    char buf[64];
    snprintf(buf, sizeof(buf), "cycle %4d", i);
    time(buf);
    
    /*
    if (foundK) {
      printf("*** K *** %lld\n", foundK);
      break;
    }
    */
  }
  cudaDeviceSynchronize();
}

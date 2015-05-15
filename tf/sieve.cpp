#include <stdio.h>
#include <assert.h>
#include <vector>

typedef unsigned char byte;
typedef unsigned long long u64;
typedef __uint128_t u128;

byte primeDelta[] = {
#include "p1M.txt"
};

#define ASIZE(a) (sizeof(a) / sizeof(a[0]))

#define NCLASS (4 * 3 * 5)

#define NBITS 500000

#define NWORDS ((NBITS - 1) / 32 + 1)

unsigned words[NWORDS];

inline int set(int &a, int x) {
  int prev = a;
  a = x;
  return prev;
}

unsigned modinv1(unsigned n, int p) {
  int d = p;
  int prevX = 1;
  int x = 0;
  while (d) {
    unsigned q = n / d;
    n = set(d, n - q * d);
    prevX = set(x, prevX - q * x);
  }
  return (prevX >= 0) ? prevX : (prevX + p);
}

unsigned modinv(int step, int p) {
  int n = step % p;
  int q = p / n;
  int d = p - q * n;
  int x = -q;
  int prevX = 1;
  
  while (d) {
    q = n / d;
    n = set(d, n - q * d);
    prevX = set(x, prevX - q * x);
  }
  unsigned ret = (prevX >= 0) ? prevX : (prevX + p);
  assert(ret == modinv1(step, p));
  return ret;
}

unsigned invTab[ASIZE(primeDelta)];

void initInvTab(int step) {
  unsigned prev = 0;
  unsigned *out = invTab;
  for (unsigned delta : primeDelta) {
    int p = prev + delta;
    prev = p;
    unsigned inv = modinv(step, p);
    *out++ = inv | (delta << 24);
  }
}

unsigned bitToClear(u128 q, unsigned prime, unsigned inv) {
  unsigned qinv = (unsigned) (q * inv % prime);
  return qinv ? prime - qinv : 0;
}

int step(int ii, int p) {
  // ii -= 32 % p;
  return (ii -= 32 % p) < 0 ? (ii + p) : ii;
}

#define BITS(p) (m##p << i##p)
// #define BIT(p) ((i##p < 32) ? (1 << i##p) : 0)
#define BIT(p) ((unsigned)(1ull << i##p))
#define STEP(p) i##p = step(i##p, p)
#define INIT(p) int i##p = bitToClear(q, p, modinv(s, p))
#define REPEAT_P32(w, s) w(3)s w(5)s w(7)s w(11)s w(13)s w(17)s w(19)s w(23)s w(29)s w(31)
#define REPEAT_P64(w, s) w(37)s w(41)s w(43)s w(47)s w(53)s w(59)s w(61)

void sieve(unsigned q, unsigned s, unsigned *pw, unsigned *end) {
  unsigned m3  = 0x49249249;
  unsigned m5  = 0x42108421;
  unsigned m7  = 0x10204081;
  unsigned m11 = 0x00400801;
  unsigned m13 = 0x04002001;
  unsigned m17 = 0x00020001;
  unsigned m19 = 0x00080001;
  unsigned m23 = 0x00800001;
  unsigned m29 = 0x20000001;
  unsigned m31 = 0x80000001;

  REPEAT_P32(INIT, ;);
  REPEAT_P64(INIT, ;);
  
  for (; pw < end; ++pw) {
    *pw = REPEAT_P32(BITS, |) | REPEAT_P64(BIT, |);
    REPEAT_P32(STEP, ;);
    REPEAT_P64(STEP, ;);
  }

  unsigned p = 0;
  for (unsigned info : invTab) {
    p += (info >> 24);
    unsigned inv = info & 0xffffff;
    unsigned btc = bitToClear(q, p, inv);
    while (btc < NBITS) {
      words[btc >> 5] |= (1 << (btc & 31));
      btc += p;
    }
  }  
}

std::vector<unsigned> extract(unsigned *pw, unsigned *end) {
  std::vector<unsigned> ret;
  int bitPos = 0;
  for (; pw < end; ++pw, bitPos += 32) {
    unsigned bits = ~*pw;
    while (bits) {
      int i = __builtin_ctz(bits);
      bits &= bits - 1;
      ret.push_back(bitPos + i);
    }
  }
  return ret;
}

std::vector<unsigned> mapPrimes(const std::vector<unsigned> &bitPos, unsigned q, unsigned s) {
  std::vector<unsigned> ret;
  ret.reserve(bitPos.size());
  for (unsigned pos : bitPos) {
    ret.push_back(q + pos * s);
  }
  return ret;
}

// whether 2 * k * p + 1 == 1 or 7 modulo 8.
bool q1or7mod8(unsigned p, u64 k) {
  return !(k & 3) || ((k & 3) + (p & 3) == 4);
}

// whether 2 * k * p + 1 != 0 modulo prime
bool notMultiple(unsigned p, unsigned k, unsigned prime) {
  // return (((k + k) % prime) * (p % prime) + 1) % prime != 0;
  unsigned kk = k % prime;
  return !kk || ((p % prime) * kk * 2 + 1) % prime != 0;
  
  // return ((p % prime) * 2 * (u64)k + 1) % prime != 0;
}

bool acceptClass(unsigned p, unsigned c) {
  return q1or7mod8(p, c) && notMultiple(p, c, 3) && notMultiple(p, c, 5);
}

int main() {
  unsigned q = 1000001;
  unsigned s = 2;
  initInvTab(s);
  sieve(q, s, words, words + NWORDS);
  auto bitPos = extract(words, words + NWORDS);
  auto primes = mapPrimes(bitPos, q, s);
  for (unsigned p : primes) {
    printf("%d\n", p);
  }

  /*
  const unsigned p = 119904229;
  int startPow2 = 67;
  u64 auxK = (((u128) 1) << (startPow2 - 1)) / p;
  u64 k0 = auxK - auxK % NCLASS;
  // u64 repeat = auxK / NCLASS + 1;
  for (int c = 0; c <= NCLASS; ++c) {
    if (accept(p, c)) {
      sieve();
    }
  }
  */
}

void print(std::vector<int> &primes) {
  int prev = 0;
  int cnt = 0;
  for (int p : primes) {
    printf("%2d, ", p - prev);
    ++cnt;
    if (cnt == 25) {
      printf("\n");
      cnt = 0;
    }
    prev = p;
  }
  printf("\n");
}

/*
u128 makeQinv(unsigned exp, u64 k, unsigned inv) {
  return 2 * exp * (u64) inv * (u128) k + inv;
}

u64 makeQinv(unsigned q, unsigned inv) {
  return q * (u64) inv;
}

unsigned bitToClearAux(u128 qinv, unsigned prime) {
  return prime - (unsigned)(qInv % prime);
}

unsigned bitToClear(unsigned exp, u64 k, unsigned prime, unsigned inv) {
  // return ((p - q % p) * (u64) inv) % p;
  // u128 qInv = (2 * k * (u128) exp + 1) * inv;
  u128 qInv = 2 * inv * k * (u128) exp + inv;
  unsigned ret = prime - (unsigned)(qInv % prime);
  assert(ret < prime);
  assert(ret == bitToClearSlow(exp, k, prime, inv));
  return ret;
}

unsigned bitToClearSlow(unsigned exp, u64 k, unsigned prime, unsigned inv) {
  unsigned kmod = k % prime;
  unsigned qmod = (2 * kmod * (u64) exp + 1) % prime;
  return (prime - qmod) * (u64)inv % prime;
}
*/

/*
int smallPrimes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61};
int primes1K[] = {
   67,    71, 
   73,    79,    83,    89,    97,   101,   103,   107,   109,   113, 
  127,   131,   137,   139,   149,   151,   157,   163,   167,   173, 
  179,   181,   191,   193,   197,   199,   211,   223,   227,   229, 
  233,   239,   241,   251,   257,   263,   269,   271,   277,   281, 
  283,   293,   307,   311,   313,   317,   331,   337,   347,   349, 
  353,   359,   367,   373,   379,   383,   389,   397,   401,   409, 
  419,   421,   431,   433,   439,   443,   449,   457,   461,   463, 
  467,   479,   487,   491,   499,   503,   509,   521,   523,   541, 
  547,   557,   563,   569,   571,   577,   587,   593,   599,   601, 
  607,   613,   617,   619,   631,   641,   643,   647,   653,   659, 
  661,   673,   677,   683,   691,   701,   709,   719,   727,   733, 
  739,   743,   751,   757,   761,   769,   773,   787,   797,   809, 
  811,   821,   823,   827,   829,   839,   853,   857,   859,   863, 
  877,   881,   883,   887,   907,   911,   919,   929,   937,   941, 
  947,   953,   967,   971,   977,   983,   991,   997
};
*/

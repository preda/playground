// Copyright (c) Mihai Preda, 2015.

#include <stdio.h>
#include <assert.h>
#include <vector>

typedef unsigned char byte;
typedef unsigned long long u64;
typedef __uint128_t u128;

#define ASIZE(a) (sizeof(a) / sizeof(a[0]))
#define BITS_TO_WORDS(b) (((b) - 1) / 32  + 1)

byte primeDelta[] = {
#include "p1M.txt"
};

inline int set(int &a, int x) {
  int prev = a;
  a = x;
  return prev;
}

unsigned modinv1(u64 step, int p) {
  unsigned n = step % p;
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

unsigned modinv(u64 step, unsigned p) {
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

unsigned bitToClear(u128 q, unsigned prime, unsigned inv) {
  unsigned qinv = (unsigned) (q * inv % prime);
  return qinv ? prime - qinv : 0;
}

unsigned invTab[ASIZE(primeDelta)];
unsigned btcTab[ASIZE(primeDelta)];

void initInvTab(u64 step) {
  unsigned prev = 0;
  unsigned *out = invTab;
  for (unsigned delta : primeDelta) {
    *out++ = modinv(step, prev += delta);
  }
}

void initBtcTab(u128 q) {
  unsigned prev = 0;
  unsigned *out = btcTab;
  unsigned *pInv = invTab;
  for (unsigned delta : primeDelta) {
    *out++ = bitToClear(q, prev += delta, *pInv++);    
  }
}

int step(int ii, int p) {
  // ii -= 32 % p;
  return (ii -= 32 % p) < 0 ? (ii + p) : ii;
}

#define BITS(p) (m##p << i##p)
// #define BIT(p) ((i##p < 32) ? (1 << i##p) : 0)
#define BIT(p) ((unsigned)(1ull << i##p))
#define STEP(p) i##p = step(i##p, p)
#define INIT(p) int i##p = *pBtc++
#define SAVE_BTC(p) *pBtc++ = i##p
// bitToClear(q, p, modinv(s, p))
// w(3)s w(5)s w(7)s 
#define REPEAT_00_32(w, s) w(11)s w(13)s w(17)s w(19)s w(23)s w(29)s w(31)
#define REPEAT_32_64(w, s) w(37)s w(41)s w(43)s w(47)s w(53)s w(59)s w(61)
#define REPEAT(w, s) REPEAT_00_32(w, s)s REPEAT_32_64(w, s)

void sieve(u64 s, unsigned *words, unsigned nWords) {
  // unsigned m3  = 0x49249249;
  // unsigned m5  = 0x42108421;
  // unsigned m7  = 0x10204081;
  unsigned m11 = 0x00400801;
  unsigned m13 = 0x04002001;
  unsigned m17 = 0x00020001;
  unsigned m19 = 0x00080001;
  unsigned m23 = 0x00800001;
  unsigned m29 = 0x20000001;
  unsigned m31 = 0x80000001;
  unsigned *pBtc;
  pBtc = btcTab;
  REPEAT(INIT, ;); // REPEAT_P32(INIT, ;); REPEAT_P64(INIT, ;);
  for (unsigned *pw = words, *end = words + nWords; pw < end; ++pw) {
    *pw = REPEAT_00_32(BITS, |) | REPEAT_32_64(BIT, |);
    REPEAT(STEP, ;);
  }
  pBtc = btcTab;
  REPEAT(SAVE_BTC, ;);
  
  unsigned skip = 14;
  pBtc = btcTab + skip;
  unsigned prime = 61;
  unsigned nBits = nWords * 32;
  unsigned *wEnd = words + nWords;
  for (byte *pDelta = primeDelta + skip, *end = primeDelta + ASIZE(primeDelta); pDelta < end; ++pDelta) {
    prime += *pDelta;
    unsigned btc = *pBtc;
    while (btc < nBits) {
      words[btc >> 5] |= (1 << (btc & 31));
      btc += prime;
    }
    *pBtc++ = btc - nBits;
  }
}

std::vector<unsigned> extract(unsigned *pw, unsigned nWords) {
  std::vector<unsigned> ret;
  int bitPos = 0;
  for (unsigned *end = pw + nWords; pw < end; ++pw, bitPos += 32) {
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
bool q1or7mod8(unsigned exp, u64 c) {
  return !(c & 3) || ((c & 3) + (exp & 3) == 4);
}

// whether 2 * k * p + 1 != 0 modulo prime
bool notMultiple(unsigned exp, unsigned c, unsigned prime) {
  // return (((k + k) % prime) * (p % prime) + 1) % prime != 0;
  // return ((p % prime) * 2 * (u64)k + 1) % prime != 0;
  return !c || (2 * c * (u64) exp + 1) % prime;
}

bool acceptClass(unsigned exp, unsigned c) {
  return q1or7mod8(exp, c) && notMultiple(exp, c, 3) && notMultiple(exp, c, 5)
    && notMultiple(exp, c, 7);
}

#define NCLASS (4 * 3 * 5 * 7)
#define MAX_WORDS (16 * 1024)
#define MAX_BITS (MAX_WORDS << 5)
unsigned words[MAX_WORDS];

int main() {
  const unsigned exp = 119904229;
  int startPow2 = 67;
  const u64 beginK = ((((u128) 1) << (startPow2 - 1)) - 1) / exp + 1;
  const u64 endK = beginK + 4000000000ll;
  // u64 endK   = (((u128) 1) << startPow2) / exp;
  printf("Initial beginK %llu endK %llu\n", beginK, endK);
  unsigned classMod = beginK % NCLASS;
  u64 step = 2 * NCLASS * (u64) exp;
  initInvTab(step);
  // u64 k0 = beginK - classMod;
  u64 total = 0;
  for (int c = 0; c <= NCLASS; ++c) {
    if (acceptClass(exp, c)) {
      u64 k = beginK + ((c < classMod) ? (NCLASS - classMod + c) : (c - classMod));
      u128 q = 2 * exp * (u128) k + 1;
      initBtcTab(q);
      int primesInClass = 0;
      while (k < endK) {          
        u64 nK = (endK - beginK - 1) / NCLASS + 1;
        unsigned nWords = (MAX_BITS < nK ) ? MAX_WORDS : BITS_TO_WORDS(nK);
        sieve(step, words, nWords);
        auto bits = extract(words, nWords); // use NCLASS
        primesInClass += bits.size();
        // printf("%3d: %lu\n", c, bits.size());
        k += NCLASS * 32 * nWords; 
      }
      printf("class %d primes %d\n", c, primesInClass);
      total += primesInClass;
    }
  }
  printf("total %llu\n", total);
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
    if (btc < nBits) {
      unsigned *pw = words + (btc >> 5);
      int bitPos = btc & 31;
      int wordStep = (prime >> 5) + 1;
      int bitStep = prime - (wordStep << 5);
      do {
        *pw |= (1 << bitPos);
        pw += wordStep;
        bitPos += bitStep;
        if (bitPos < 0) {
          --pw;
          bitPos += 32;
        }
      } while (pw < wEnd);
      *pBtc++ = ((pw - words) << 5) + bitPos - nBits;
    } else {
      *pBtc++ = btc - nBits;
    }


  unsigned q = 1000001;
  unsigned s = 2;
  initInvTab(s);
  sieve(q, s, words, NWORDS);
  auto bitPos = extract(words, words + NWORDS);
  auto primes = mapPrimes(bitPos, q, s);
  for (unsigned p : primes) {
    printf("%d\n", p);
  }

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

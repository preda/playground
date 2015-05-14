#include <stdio.h>
#include <assert.h>
#include <vector>

// #define NCLASS (2 * 3 * 5 * 7)

// int smallPrimes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61};

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

#define ASIZE(a) (sizeof(a) / sizeof(a[0]))

#define NBITS 500000

#define NWORDS ((NBITS - 1) / 32 + 1)

unsigned words[NWORDS];

inline int set(int &a, int x) {
  int prev = a;
  a = x;
  return prev;
}

unsigned modinv(unsigned n, int p) {
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

unsigned bitToClear(unsigned c, unsigned step, unsigned p) {
  assert(c > 0);
  return ((p - c % p) * modinv(step, p)) % p;
}

int step(int ii, int p) {
  ii -= 32 % p;
  return ii < 0 ? (ii + p) : ii;
}

#define BITS(p) (m##p << i##p)
// #define BIT(p) ((i##p < 32) ? (1 << i##p) : 0)
#define BIT(p) ((unsigned)(1ull << i##p))
#define STEP(p) i##p = step(i##p, p)
#define INIT(p) int i##p = bitToClear(c, s, p)
#define REPEAT_P32(w, s) w(3)s w(5)s w(7)s w(11)s w(13)s w(17)s w(19)s w(23)s w(29)s w(31)
#define REPEAT_P64(w, s) w(37)s w(41)s w(43)s w(47)s w(53)s w(59)s w(61)

void sieve(unsigned c, unsigned s) {
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
  
  for (unsigned *pw = words, *end = words + NWORDS; pw < end; ++pw) {
    *pw = REPEAT_P32(BITS, |) | REPEAT_P64(BIT, |);
    REPEAT_P32(STEP, ;);
    REPEAT_P64(STEP, ;);
  }

  words[0] |= 1;
  
  for (int *pp = primes1K, *end = primes1K + sizeof(primes1K) / sizeof(primes1K[0]); pp < end; ++pp) {
    int p = *pp;
    int btc = bitToClear(c, s, p);
    while (btc < NBITS) {
      words[btc >> 5] |= (1 << (btc & 31));
      btc += p;
    }
  }
  
}

void extract(std::vector<int> &primes, int c, int s, unsigned *pw, unsigned *end) {
  for (; pw < end; ++pw, c += 32 * s) {
    unsigned bits = ~*pw;
    while (bits) {
      int i = __builtin_ctz(bits);
      bits &= bits - 1;
      int p = c + i * s;
      primes.push_back(p);
    }
  }
}

int main() {
  sieve(1, 2);
  std::vector<int> primes(primes1K, primes1K + ASIZE(primes1K));
  extract(primes, 1, 2, words, words + NWORDS);
  // printf("N %lu\n", primes.size());
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

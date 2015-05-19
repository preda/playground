typedef unsigned long long u64;
typedef unsigned short u16;
typedef __uint128_t u128;
typedef unsigned char byte;

// whether 2 * c * p + 1 == 1 or 7 modulo 8.
bool q1or7mod8(unsigned exp, u64 c) {
  return !(c & 3) || ((c & 3) + (exp & 3) == 4);
}

// whether 2 * k * p + 1 != 0 modulo prime
bool notMultiple(unsigned exp, unsigned c, unsigned prime) {
  return !c || (2 * c * (u64) exp + 1) % prime;
}

bool acceptClass(unsigned exp, unsigned c) {
  return q1or7mod8(exp, c) && notMultiple(exp, c, 3) && notMultiple(exp, c, 5) && notMultiple(exp, c, 7);
  /*
    && notMultiple(p, k, 11) && notMultiple(p, k, 13) && notMultiple(p, k, 17)
    && notMultiple(p, k, 19) && notMultiple(p, k, 23);
  */
}






/*
// whether 2 * k * p + 1 != 0 modulo prime
extern inline bool notMultiple(unsigned p, unsigned k, unsigned prime) {
  unsigned kk = k % prime;
  return !kk || ((p % prime) * kk * 2 + 1) % prime != 0;
  // return (((k + k) % prime) * (p % prime) + 1) % prime != 0;
  // return ((p % prime) * 2 * (u64)k + 1) % prime != 0;
}
*/

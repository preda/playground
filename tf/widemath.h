typedef unsigned char u8;
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

#define DEVICE __device__ static

DEVICE U6 _U6(U3 x)  { return (U6) {x.a, x.b, x.c, 0, 0, 0}; }
DEVICE U5 _U5(U4 x)  { return (U5) {x.a, x.b, x.c, x.d, 0}; }
DEVICE U6 _U6(U5 x)  { return (U6) {x.a, x.b, x.c, x.d, x.e, 0}; }
DEVICE U6 _U6(U4 x)  { return _U6(_U5(x)); }
DEVICE U2 _U2(u64 x) { return (U2) {(u32) x, (u32) (x >> 32)}; }

DEVICE U5 operator+(U5 x, U3 y) {
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

DEVICE U4 operator+(U4 x, U3 y) {
  u32 a, b, c, d;
  asm("add.cc.u32  %0, %4, %8;"
      "addc.cc.u32 %1, %5, %9;"
      "addc.cc.u32 %2, %6, %10;"
      "addc.u32    %3, %7, 0;"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(x.d), "r"(y.a), "r"(y.b), "r"(y.c));
  return (U4) {a, b, c, d};
}

DEVICE U3 operator+(U3 x, U3 y) {
  u32 a, b, c;
  asm("add.cc.u32  %0, %3, %6;"
      "addc.cc.u32 %1, %4, %7;"
      "addc.u32    %2, %5, %8;"
      : "=r"(a), "=r"(b), "=r"(c)
      : "r"(x.a), "r"(x.b), "r"(x.c),
        "r"(y.a), "r"(y.b), "r"(y.c));
  return (U3) {a, b, c};
}

DEVICE U4 operator-(U4 x, U4 y) {
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

DEVICE U5 operator-(U5 x, U5 y) {
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

// 4 MULs.
DEVICE U3 operator*(U2 x, u32 n) {
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

// compute x * n + 1
DEVICE U3 incMul(U2 x, u32 n) {
  u32 a, b, c;
  asm(
      "mul.hi.u32     %1, %3, %5;"
      "mad.lo.cc.u32  %0, %3, %5, 1;"
      "madc.lo.cc.u32 %1, %4, %5, %1;"
      "madc.hi.u32    %2, %4, %5, 0;"
      : "=r"(a), "=r"(b), "=r"(c)
      : "r"(x.a), "r"(x.b), "r"(n));
  return (U3) {a, b, c};
}

// 6 MULs.
DEVICE U4 operator*(U3 x, u32 n) {
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

DEVICE U5 shr1w(U6 x) { return (U5) {x.b, x.c, x.d, x.e, x.f}; }
DEVICE U4 shr1w(U5 x) { return (U4) {x.b, x.c, x.d, x.e}; }
DEVICE U3 shr1w(U4 x) { return (U3) {x.b, x.c, x.d}; }

// Funnel shift left.
DEVICE u32 shl(u32 a, u32 b, int n) {
  u32 r;
  asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(n));
  return r;
}

// Funnel shift right.
DEVICE u32 shr(u32 a, u32 b, int n) {
  u32 r;
  asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(n));
  return r;
}

DEVICE U3 operator<<(U3 x, int n) {
  assert(n >= 0 && n < 32 && !(x.c >> (32 - n)));
  return (U3) {x.a << n, shl(x.a, x.b, n), shl(x.b, x.c, n)};
}

DEVICE U4 operator<<(U4 x, int n) {
  // assert(n >= 0 && n < 32 && !(x.d >> (32 - n)));
  return (U4) {x.a << n, shl(x.a, x.b, n), shl(x.b, x.c, n), shl(x.c, x.d, n)};
}

DEVICE U5 operator<<(U5 x, int n) {
  assert(n >= 0 && n < 32 && !(x.e >> (32 - n)));
  U4 t = (U4) {x.a, x.b, x.c, x.d} << n;
  return (U5) {t.a, t.b, t.c, t.d, shl(x.d, x.e, n)};
}

DEVICE void operator-=(U4 &x, U4 y) { x = x - y; }

DEVICE void operator<<=(U3 &x, int n) { x = x << n; }

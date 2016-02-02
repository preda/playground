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

DEVICE U6 _U6(U3 x)  { return (U6) {x.a, x.b, x.c, 0, 0, 0}; }
DEVICE U5 _U5(U4 x)  { return (U5) {x.a, x.b, x.c, x.d, 0}; }
DEVICE U4 _U4(U3 x)  { return (U4) {x.a, x.b, x.c, 0}; }
DEVICE U6 _U6(U5 x)  { return (U6) {x.a, x.b, x.c, x.d, x.e, 0}; }
DEVICE U6 _U6(U4 x)  { return _U6(_U5(x)); }
DEVICE U2 _U2(u64 x) { return (U2) {(u32) x, (u32) (x >> 32)}; }
DEVICE u64 _u64(U2 x) { return (((u64) x.b) << 32) | x.a; }

DEVICE U5 shr1w(U6 x) { return (U5) {x.b, x.c, x.d, x.e, x.f}; }
DEVICE U4 shr1w(U5 x) { return (U4) {x.b, x.c, x.d, x.e}; }
DEVICE U3 shr1w(U4 x) { return (U3) {x.b, x.c, x.d}; }
DEVICE U2 shr1w(U3 x) { return (U2) {x.b, x.c}; }
DEVICE U4 shl1w(U3 x) { return (U4) {0, x.a, x.b, x.c}; }

DEVICE U3 operator~(U3 x) {
  return (U3) {~x.a, ~x.b, ~x.c};
}

DEVICE bool operator==(U3 x, U3 y) {
  return x.a == y.a && x.b == y.b && x.c == y.c;
}

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

DEVICE U3 operator+(U3 x, U2 y) {
  u32 a, b, c;
  asm("add.cc.u32  %0, %3, %6;"
      "addc.cc.u32 %1, %4, %7;"
      "addc.u32    %2, %5, 0;"
      : "=r"(a), "=r"(b), "=r"(c)
      : "r"(x.a), "r"(x.b), "r"(x.c),
        "r"(y.a), "r"(y.b));
  return (U3) {a, b, c};
}

DEVICE U3 operator+(U3 x, u32 y) {
  u32 a, b, c;
  asm("add.cc.u32  %0, %3, %6;"
      "addc.cc.u32 %1, %4, 0;"
      "addc.u32    %2, %5, 0;"
      : "=r"(a), "=r"(b), "=r"(c)
      : "r"(x.a), "r"(x.b), "r"(x.c),
        "r"(y));
  return (U3) {a, b, c};
}

DEVICE U2 operator+(U2 x, u32 y) {
  u32 a, b;
  asm("add.cc.u32  %0, %2, %4;"
      "addc.u32 %1, %3, 0;"
      : "=r"(a), "=r"(b)
      : "r"(x.a), "r"(x.b),
        "r"(y));
  return (U2) {a, b};
}

DEVICE U3 operator-(U3 x, U3 y) {
  u32 a, b, c;
  asm("sub.cc.u32  %0, %3, %6;"
      "subc.cc.u32 %1, %4, %7;"
      "subc.u32    %2, %5, %8;"
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

// 5 MULs.
DEVICE U3 mulLow(U3 x, u32 n) {
  u32 a, b, c;
  asm(
      "mul.hi.u32     %1, %3, %6;"
      "mul.lo.u32     %2, %5, %6;"
      "mad.lo.cc.u32  %1, %4, %6, %1;"
      "mul.lo.u32     %0, %3, %6;"
      "madc.hi.u32    %2, %4, %6, %2;"
      : "=r"(a), "=r"(b), "=r"(c)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(n));
  return (U3) {a, b, c};
}

// 9 MULs
DEVICE U3 mulLow(U3 x, U3 y) {
  u32 a, b, c;
  asm(
      "mul.lo.u32    %2, %5, %6;"
      "mul.lo.u32    %0, %3, %6;"
      "mad.lo.u32    %2, %3, %8, %2;"      
      "mul.hi.u32    %1, %3, %6;"
      "mad.lo.u32    %2, %4, %7, %2;"
      "mad.lo.cc.u32 %1, %4, %6, %1;"
      "madc.hi.u32   %2, %4, %6, %2;"
      "mad.lo.cc.u32 %1, %3, %7, %1;"
      "madc.hi.u32   %2, %3, %7, %2;"
      : "=r"(a), "=r"(b), "=r"(c)
      : "r"(x.a), "r"(x.b), "r"(x.c),
        "r"(y.a), "r"(y.b), "r"(y.c));
  return (U3) {a, b, c};
}

// 9 MULs. Disregards carry from lower (not computed) digits, thus has an error of at most 4.
DEVICE U3 shr3wMul(U3 x, U3 y) {
  u32 a, b, c;
  asm(
      "mul.hi.u32     %0, %5, %6;"
      "mad.lo.cc.u32  %0, %5, %7, %0;"
      "addc.u32       %1, 0, 0;"

      "mad.lo.cc.u32  %0, %4, %8, %0;"
      "madc.hi.u32    %1, %5, %7, %1;"

      "mad.hi.cc.u32  %0, %3, %8, %0;"
      "madc.lo.cc.u32 %1, %5, %8, %1;"
      "madc.hi.u32    %2, %5, %8, 0;"

      "mad.hi.cc.u32  %0, %4, %7, %0;"
      "madc.hi.cc.u32 %1, %4, %8, %1;"
      "addc.u32       %2, %2, 0;"
      : "=r"(a), "=r"(b), "=r"(c)
      : "r"(x.a), "r"(x.b), "r"(x.c),
        "r"(y.a), "r"(y.b), "r"(y.c));
  return (U3) {a, b, c};
}

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

// Computes x * x; 6 MULs.
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

// Computes x * x; 11 MULs. x at most 80bits. 
DEVICE U5 square(U3 x) {
  assert(!(x.c & 0xffff0000));
  U2 ab = {x.a, x.b};
  U4 ab2 = square(ab);
  U3 abc = ab * (x.c + x.c) + (U3) {ab2.c, ab2.d, x.c * x.c};
  return (U5) {ab2.a, ab2.b, abc.a, abc.b, abc.c};
}
  /*
  u32 c, d, e;
  asm(
      "add.cc.u32  %0, %3, %5;"
      "addc.cc.u32 %1, %4, %6;"
      "madc.lo.u32 %2, %8, %8, %7;"
      : "=r"(c), "=r"(d), "=r"(e)
      : "r"(ab2.c), "r"(ab2.d), "r"(abc.a), "r"(abc.b), "r"(abc.c), "r"(x.c));
  assert(!(e & 0xc0000000));
  return (U5) {ab2.a, ab2.b, c, d, e};
  */


/*
DEVICE U5 square(U3 x) {
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

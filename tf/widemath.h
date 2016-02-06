DEVICE U5 _U5(U4 x)  { return (U5) {x.a, x.b, x.c, x.d, 0}; }
DEVICE U4 _U4(U3 x)  { return (U4) {x.a, x.b, x.c, 0}; }
DEVICE U2 _U2(u64 x) { return (U2) {(u32) x, (u32) (x >> 32)}; }
DEVICE u64 _u64(U2 x) { return (((u64) x.b) << 32) | x.a; }

DEVICE U4 shr1w(U5 x) { return (U4) {x.b, x.c, x.d, x.e}; }
DEVICE U3 shr1w(U4 x) { return (U3) {x.b, x.c, x.d}; }
DEVICE U2 shr1w(U3 x) { return (U2) {x.b, x.c}; }
DEVICE U4 shl1w(U3 x) { return (U4) {0, x.a, x.b, x.c}; }

DEVICE U3 operator~(U3 x) { return (U3) {~x.a, ~x.b, ~x.c}; }
DEVICE bool operator==(U3 x, U3 y) { return x.a == y.a && x.b == y.b && x.c == y.c; }

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

// Lower 3 words of x * n; 5 MULs.
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

// Lower 3 words of x * y; 9 MULs.
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

// (x * y) >> 96 (upper 3 words of x * y); 9 MULs.
// Disregards carry from lower (not computed) digits, thus has an error of at most 4.
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

DEVICE U3 operator<<(U3 x, int n) {
  assert(n >= 0 && n < 32 && !(x.c >> (32 - n)));
  return (U3) {x.a << n, shl(x.a, x.b, n), shl(x.b, x.c, n)};
}

DEVICE U4 operator<<(U4 x, int n) {
  assert(n >= 0 && n < 32 && !(x.d >> (32 - n)));
  return (U4) {x.a << n, shl(x.a, x.b, n), shl(x.b, x.c, n), shl(x.c, x.d, n)};
}

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

// Computes x * x, x at most 80bits; 11 MULs. 
DEVICE U5 square(U3 x) {
  assert(!(x.c & 0xffff0000));
  U2 ab = {x.a, x.b};
  U4 ab2 = square(ab);
  U3 abc = ab * (x.c + x.c) + (U3) {ab2.c, ab2.d, x.c * x.c};
  return (U5) {ab2.a, ab2.b, abc.a, abc.b, abc.c};
}

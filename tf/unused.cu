struct U6 { u32 a, b, c, d, e, f; };

DEVICE U6 _U6(U3 x)  { return (U6) {x.a, x.b, x.c, 0, 0, 0}; }
DEVICE U6 _U6(U5 x)  { return (U6) {x.a, x.b, x.c, x.d, x.e, 0}; }
DEVICE U6 _U6(U4 x)  { return _U6(_U5(x)); }
DEVICE U5 shr1w(U6 x) { return (U5) {x.b, x.c, x.d, x.e, x.f}; }

DEVICE U5 operator<<(U5 x, int n) {
  assert(n >= 0 && n < 32 && !(x.e >> (32 - n)));
  U4 t = (U4) {x.a, x.b, x.c, x.d} << n;
  return (U5) {t.a, t.b, t.c, t.d, shl(x.d, x.e, n)};
}

DEVICE void operator-=(U4 &x, U4 y) { x = x - y; }

DEVICE void operator<<=(U3 &x, int n) { x = x << n; }

// Funnel shift right.
DEVICE u32 shr(u32 a, u32 b, int n) {
  u32 r;
  asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(n));
  return r;
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

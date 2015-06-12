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

__device__ U5 operator+(U5 x, U3 y) {
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

__device__ U4 operator+(U4 x, U3 y) {
 u32 a, b, c, d;
  asm("add.cc.u32  %0, %4, %8;"
      "addc.cc.u32 %1, %5, %9;"
      "addc.cc.u32 %2, %6, %10;"
      "addc.u32    %3, %7, 0;"
      : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
      : "r"(x.a), "r"(x.b), "r"(x.c), "r"(x.d), "r"(y.a), "r"(y.b), "r"(y.c));
  return (U4) {a, b, c, d};
}

__device__ U3 operator+(U3 x, U3 y) {
  u32 a, b, c;
  asm("add.cc.u32  %0, %3, %6;"
      "addc.cc.u32 %1, %4, %7;"
      "addc.u32    %2, %5, %8;"
      : "=r"(a), "=r"(b), "=r"(c)
      : "r"(x.a), "r"(x.b), "r"(x.c),
        "r"(y.a), "r"(y.b), "r"(y.c));
  return (U3) {a, b, c};
}

__device__ static U4 operator-(U4 x, U4 y) {
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

__device__ static U5 operator-(U5 x, U5 y) {
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
__device__ U3 operator*(U2 x, u32 n) {
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
__device__ U4 operator*(U3 x, u32 n) {
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

__device__ U5 shr1w(U6 x) { return (U5) {x.b, x.c, x.d, x.e, x.f}; }
__device__ U4 shr1w(U5 x) { return (U4) {x.b, x.c, x.d, x.e}; }
__device__ U3 shr1w(U4 x) { return (U3) {x.b, x.c, x.d}; }

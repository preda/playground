
#define FUNCS(T) \
\
T##2 __attribute__((overloadable)) shift(T##2 a, int e) {             \
  switch (e) {\
  case -1: return (T##2) (a.y, -a.x);\
  case  1: return (T##2) (-a.y, a.x);\
  default: return 0;\
  }\
}\
\
T##4 __attribute__((overloadable)) shift(T##4 a, int e) {\
  switch (e) {\
  case -3: return (T##4) (a.w, -a.xyz);\
  case -2: return (T##4) (a.zw, -a.xy);\
  case -1: return (T##4) (a.yzw, -a.x);\
  case  1: return (T##4) (-a.w, a.xyz);\
  case  2: return (T##4) (-a.zw, a.xy);\
  case  3: return (T##4) (-a.yzw, a.x);\
  default: return 0;\
  }\
}\
\
T __attribute__((overloadable)) halfAdd(T x, T y) { return (x >> 1) + (y >> 1) + (x & 1); } \
\
T __attribute__((overloadable)) readC(global T *in, int W, int line, int p) {\
  T u = READ(in, W, line, p & (W - 1));                                  \
  return (p < W) ? u : -u;\
}\
\
void __attribute__((overloadable)) writeC(T u, global T *out, int W, int line, int p) {\
  WRITE((p < W) ? u : -u, out, W, line, p & (W - 1));                    \
}\
\
T##2 __attribute__((overloadable)) read2(global T *in, int W, int line, int p) {\
  return (T##2) (READ(in, W, line, p), readC(in, W, line, p + W / 2)); \
}\
\
T##2 __attribute__((overloadable)) read2C(global T *in, int W, int line, int p) {\
  return (T##2) (readC(in, W, line, p), readC(in, W, line, p + W / 2));\
}\
\
void __attribute__((overloadable)) write2(T##2 u, global T *out, int W, int line, int p) {\
  WRITE(u.x, out, W, line, p);\
  writeC(u.y, out, W, line, p + W / 2);    \
}\
\
void __attribute__((overloadable)) write2C(T##2 u, global T *out, int W, int line, int p) {\
  writeC(u.x, out, W, line, p);\
  writeC(u.y, out, W, line, (p + W / 2) & (W * 2 - 1)); \
}\

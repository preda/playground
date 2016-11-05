
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
T __attribute__((overloadable)) read(global T *in, int N, int line, int p) {\
  return in[cut((line << N) + p)];                                      \
}\
\
T __attribute__((overloadable)) readC(global T *in, int N, int line, int p) {\
  T u = read(in, N, line, p & ((1 << N) - 1));\
  return (p < (1 << N)) ? u : -u;\
}\
\
void __attribute__((overloadable)) write(T u, global T *out, int N, int line, int p) {\
  out[cut((line << N) + p)] = u;                                        \
}\
\
void __attribute__((overloadable)) writeC(T u, global T *out, int N, int line, int p) {\
  write((p < (1 << N)) ? u : -u, out, N, line, p & ((1 << N) - 1)); \
}\
\
T##2 __attribute__((overloadable)) read2(global T *in, int N, int line, int p) {\
  return (T##2) (read(in, N, line, p), readC(in, N, line, p + (1 << (N - 1)))); \
}\
\
T##2 __attribute__((overloadable)) read2C(global T *in, int N, int line, int p) {\
  return (T##2) (readC(in, N, line, p), readC(in, N, line, p + (1 << (N - 1))));\
}\
\
void __attribute__((overloadable)) write2(T##2 u, global T *out, int N, int line, int p) {\
  write(u.x, out, N, line, p);\
  writeC(u.y, out, N, line, p + (1 << (N - 1)));    \
}\
\
void __attribute__((overloadable)) write2C(T##2 u, global T *out, int N, int line, int p) {\
  writeC(u.x, out, N, line, p);\
  writeC(u.y, out, N, line, (p + (1 << (N - 1))) & ((1 << (N + 1)) - 1)); \
}\
\
T##4 __attribute__((overloadable)) addsub(T##2 a, T##2 b) { return (T##4) (a + b, a - b); }\
\

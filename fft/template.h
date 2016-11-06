
#define FUNCS(T) \
\
T##2 OVERLOADED shift(T##2 a, int e) {             \
  switch (e) {\
  case -1: return (T##2) (a.y, -a.x);\
  case  1: return (T##2) (-a.y, a.x);\
  default: return 0;\
  }\
}\
\
T##4 OVERLOADED shift(T##4 a, int e) {\
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
T OVERLOADED halfAdd(T x, T y) { return (x >> 1) + (y >> 1) + (x & 1); } \
T OVERLOADED read(global T *in, uint N, uint line, uint p) { return in[cut((line << N) + p)]; } \
void OVERLOADED write(T u, global T *out, uint N, uint line, uint p) { out[cut((line << N) + p)] = u; } \
T OVERLOADED readC(global T *in, uint N, uint line, uint p) {\
  T u = read(in, N, line, p & ((1 << N) - 1));                          \
  return (p & (1 << N)) ? -u : u;                                       \
}\
\
void OVERLOADED writeC(T u, global T *out, uint N, uint line, uint p) {\
  write((p & (1 << N)) ? -u : u, out, N, line, p & ((1 << N) - 1));     \
}\
\
T##2 OVERLOADED read2(global T *in, uint N, uint line, uint p) {\
  return (T##2) (read(in, N, line, p), readC(in, N, line, p + (1 << (N - 1)))); \
}\
\
T##2 OVERLOADED read2C(global T *in, uint N, uint line, uint p) {\
  return (T##2) (readC(in, N, line, p), readC(in, N, line, p + (1 << (N - 1)))); \
}\
\
void OVERLOADED write2(T##2 u, global T *out, uint N, uint line, uint p) {\
  write(u.x, out, N, line, p);\
  writeC(u.y, out, N, line, p + (1 << (N - 1)));    \
}\
\
void OVERLOADED write2C(T##2 u, global T *out, uint N, uint line, uint p) {\
  writeC(u.x, out, N, line, p);\
  writeC(u.y, out, N, line, p + (1 << (N - 1)));    \
}\

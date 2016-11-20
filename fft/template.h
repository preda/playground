
#define FUNCS_SHIFT(T) \
\
T##2 _OVL shift(T##2 a, int e) { \
  switch (e) {\
  case -1: return (T##2) (a.y, -a.x);\
  case  1: return (T##2) (-a.y, a.x);\
  default: return 0;\
  }\
}\
\
T##4 _OVL shift(T##4 a, int e) {\
  switch (e) {\
  case -3: return (T##4) (a.w, -a.xyz);\
  case -2: return (T##4) (a.zw, -a.xy);\
  case -1: return (T##4) (a.yzw, -a.x);\
  case  1: return (T##4) (-a.w, a.xyz);\
  case  2: return (T##4) (-a.zw, a.xy);\
  case  3: return (T##4) (-a.yzw, a.x);\
  default: return 0;\
  }\
}

#define FUNCS_RW(T, global) \
\
T _OVL readC(global T *in, uint N, uint line, uint p) {\
  T u = read(in, N, line, p & ((1 << N) - 1));                          \
  return (p & (1 << N)) ? -u : u;                                       \
}\
\
T _OVL readT(global T *in, uint N, uint line, uint p) {\
  T u = read(in, N, line & ((1 << N) - 1), p);                      \
  return (line & (1 << N)) ? -u : u;                                       \
}\
\
void _OVL writeC(T u, global T *out, uint N, uint line, uint p) {\
  write((p & (1 << N)) ? -u : u, out, N, line, p & ((1 << N) - 1));     \
}\
\
void _OVL writeT(T u, global T *out, uint N, uint line, uint p) {\
  write((line & (1 << N)) ? -u : u, out, N, line & ((1 << N) - 1), p); \
}\
\
T##2 _OVL read2(global T *in, uint N, uint line, uint p) {\
  return (T##2) (read(in, N, line, p), readC(in, N, line, p + (1 << (N - 1)))); \
}\
\
T##2 _OVL read2C(global T *in, uint N, uint line, uint p) {\
  return (T##2) (readC(in, N, line, p), readC(in, N, line, p + (1 << (N - 1)))); \
}\
\
void _OVL write2(T##2 u, global T *out, uint N, uint line, uint p) {\
  write(u.x, out, N, line, p);\
  writeC(u.y, out, N, line, p + (1 << (N - 1)));    \
}\
\
void _OVL write2C(T##2 u, global T *out, uint N, uint line, uint p) {\
  writeC(u.x, out, N, line, p);\
  writeC(u.y, out, N, line, p + (1 << (N - 1)));    \
}

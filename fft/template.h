
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
T __attribute__((overloadable)) halfAdd(T x, T y) { return (x >> 1) + (y >> 1) + (x & 1); }\
\
T __attribute__((overloadable)) read(global T *in, int width, int line, int p) {\
  return in[cut(line * width + p)];\
}\
\
void __attribute__((overloadable)) write(T u, global T *out, int width, int line, int p) {\
  out[cut(line * width + p)] = u;\
}\


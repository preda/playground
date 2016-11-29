#define _O __attribute__((overloadable))
#define GS 256
#define KERNEL(groupSize) kernel __attribute__((reqd_work_group_size(groupSize, 1, 1)))

double2 _O shift(double2 a, int e) {
  switch (e) {
  case -1: return (double2) (a.y, -a.x);
  case  1: return (double2) (-a.y, a.x);
  default: return 0;
  }
}

double4 _O shift(double4 a, int e) {
  switch (e) {
  case -3: return (double4) (a.w, -a.xyz);
  case -2: return (double4) (a.zw, -a.xy);
  case -1: return (double4) (a.yzw, -a.x);
  case  1: return (double4) (-a.w, a.xyz);
  case  2: return (double4) (-a.zw, a.xy);
  case  3: return (double4) (-a.yzw, a.x);
  default: return 0;
  }
}

void bar() { barrier(CLK_LOCAL_MEM_FENCE); }
unsigned cut8(unsigned x) { return x & 0x1fffffff; }

double _O read(global double *in, uint N, uint line, uint p) { return in[cut8((line << N) + p)]; }
double _O read(local  double *in, uint N, uint line, uint p) { return in[(line << N) + p]; }
void _O write(double u, global double *out, uint N, uint line, uint p) { out[cut8((line << N) + p)] = u; }
void _O write(double u, local  double *out, uint N, uint line, uint p) { out[(line << N) + p] = u; }

double _O readC(global double *in, uint N, uint line, uint p) {
  double u = read(in, N, line, p & ((1 << N) - 1));
  return (p & (1 << N)) ? -u : u;
}

double _O readT(global double *in, uint N, uint line, uint p) {
  double u = read(in, N, line & ((1 << N) - 1), p);
  return (line & (1 << N)) ? -u : u;
}

void _O writeC(double u, global double *out, uint N, uint line, uint p) {
  write((p & (1 << N)) ? -u : u, out, N, line, p & ((1 << N) - 1));
}

void _O writeT(double u, global double *out, uint N, uint line, uint p) {
  write((line & (1 << N)) ? -u : u, out, N, line & ((1 << N) - 1), p);
}

double2 _O read2(global double *in, uint N, uint line, uint p) {
  return (double2) (read(in, N, line, p), readC(in, N, line, p + (1 << (N - 1))));
}

double2 _O read2C(global double *in, uint N, uint line, uint p) {
  return (double2) (readC(in, N, line, p), readC(in, N, line, p + (1 << (N - 1))));
}

void _O write2(double2 u, global double *out, uint N, uint line, uint p) {
  write(u.x, out, N, line, p);
  writeC(u.y, out, N, line, p + (1 << (N - 1)));
}

void _O write2C(double2 u, global double *out, uint N, uint line, uint p) {
  writeC(u.x, out, N, line, p);
  writeC(u.y, out, N, line, p + (1 << (N - 1)));
}

#define ADDSUB4(a, b) { double4 tmp = a; a = tmp + b; b = tmp - b; }
#define SHIFT(u, e) u = shift(u, e);

#define FFT_SETUP_CORE(N, round, radixExp, iniG) \
uint g = iniG; \
uint mr = 1 << (round * radixExp);\
uint j = g & (mr - 1);\
uint r = (g & ~(mr - 1)) << radixExp;\
uint e = j << (N + 1 - (round + 1) * radixExp);\
uint line = j + r;

#define FFT_SETUP(iniN, radixExp) \
uint N = iniN; \
uint groupsPerLine = (1 << (N - (radixExp - 1))) / GS;  \
FFT_SETUP_CORE(N, round, radixExp, get_group_id(0) / groupsPerLine);    \
uint k = get_group_id(0) & (groupsPerLine - 1);\
uint p = get_local_id(0) + k * GS;

#define QW (1 << (N - 2))

double4 _O read4(global double *in, uint N, uint line, uint p) {
  double u[4];
  for (int i = 0; i < 4; ++i) { u[i] = readC(in, N, line, p + QW * i); }
  return (double4)(u[0], u[1], u[2], u[3]);
}

double4 _O read4NC(global double *in, uint N, uint line, uint p) {
  double u[4];
  for (int i = 0; i < 4; ++i) { u[i] = read(in, N, line, p + QW * i); }
  return (double4)(u[0], u[1], u[2], u[3]);
}

void _O write4(double4 u, global double *out, uint N, uint line, uint p) {
  for (int i = 0; i < 4; ++i) { writeC((double[4]){u.x, u.y, u.z, u.w}[i], out, N, line, p + QW * i); }
}

void _O write4NC(double4 u, global double *out, uint N, uint line, uint p) {
  for (int i = 0; i < 4; ++i) { write((double[4]){u.x, u.y, u.z, u.w}[i], out, N, line, p + QW * i); }
}

#undef QW

// Radix-8 DIF step. round is 3 to 0.
KERNEL(GS) void dif8(int round, global double *in, global double *out) {
  FFT_SETUP(11, 3);
  double4 u[8];
  uint revbin[8] = {0, 4, 2, 6, 1, 5, 3, 7};

  for (int i = 0; i < 8; ++i) { u[i] = read4(in, N, line + mr * i, p); }
  for (int i = 0; i < 4; ++i) { ADDSUB4(u[i], u[i + 4]); }
  for (int i = 1; i < 4; ++i) { SHIFT(u[i + 4], i); }

  for (int i = 0; i < 8; i += 4) {
    ADDSUB4(u[0 + i], u[2 + i]);
    ADDSUB4(u[1 + i], u[3 + i]);
    SHIFT(u[3 + i], 2);
  }

  for (int i = 0; i < 8; i += 2) { ADDSUB4(u[i], u[i + 1]); }
  for (int i = 0; i < 8; ++i) { write4(u[i], out, N, line + mr * i, p + e * revbin[i]); }

  /*
  for (int i = 0; i < 8; i += 2) {
    write4(u[i] + u[i + 1], out, N, line + mr * i,       p + e * revbin[i]);
    write4(u[i] - u[i + 1], out, N, line + mr * (i + 1), p + e * revbin[i + 1]);
  }
  */
}

// Radix-8 DIT step. round is 0 to 3.
KERNEL(GS) void dit8(int round, global double *in, global double *out) {
  FFT_SETUP(11, 3);
  double4 u[8];
  uint revbin[8] = {0, 4, 2, 6, 1, 5, 3, 7};

  u[0] = read4NC(in, N, line, p);
  for (int i = 1; i < 8; ++i) { u[i] = read4(in, N, line + mr * revbin[i], p + e * i); }
  for (int i = 0; i < 4; ++i) { ADDSUB4(u[i], u[i + 4]); }
  for (int i = 1; i < 4; ++i) { SHIFT(u[i + 4], -i); }

  for (int i = 0; i < 8; i += 4) {
    ADDSUB4(u[0 + i], u[2 + i]);
    ADDSUB4(u[1 + i], u[3 + i]);
    SHIFT(u[3 + i], -2);
  }

  for (int i = 0; i < 8; i += 2) { ADDSUB4(u[i], u[i + 1]); }
  for (int i = 0; i < 8; ++i) { write4NC(u[i], out, N, line + mr * revbin[i], p); }
}


KERNEL(256) void mul(global double *tab) {
  // for (int i = 0; i < 4; ++i) { prefetch(tab + cut8(get_group_id(0) * 1024 + i * 256 + get_local_id(0)), 1); }
  
  for (int i = 0; i < 4; ++i) {
    uint p = get_global_id(0) + get_global_size(0) * i;
    // get_group_id(0) * 1024 + i * 256 + get_local_id(0);
    double x = tab[cut8(p)];
    tab[cut8(p)] = x * x + x * p;
  }
}
